package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

const (
	antigravityTranscriptionDefaultModel = "gemini-3-flash-agent"
	antigravityTranscriptionMaxResponse  = 8 << 20
)

// These names deliberately opt into the Antigravity OAuth path. OpenAI/Codex
// model names remain routed to the Codex adapter by transcriptionHandler.
var supportedAntigravityTranscriptionModels = map[string]string{
	"gemini-transcribe":          antigravityTranscriptionDefaultModel,
	"google-transcribe":          antigravityTranscriptionDefaultModel,
	"gemini-3-flash-agent":       "gemini-3-flash-agent",
	"gemini-3-flash":             "gemini-3-flash",
	"gemini-3.1-flash-lite":      "gemini-3.1-flash-lite",
	"gemini-3.5-flash-low":       "gemini-3.5-flash-low",
	"gemini-3.5-flash-extra-low": "gemini-3.5-flash-extra-low",
	"gemini-3.6-flash-high":      "gemini-3.6-flash-high",
	"gemini-3.7-flash-high":      "gemini-3.7-flash-high",
}

type transcriptionHandler struct {
	codex       *codexTranscriptionHandler
	antigravity *antigravityTranscriptionHandler
}

func newTranscriptionHandler(authManager *auth.Manager) *transcriptionHandler {
	return &transcriptionHandler{
		codex:       newCodexTranscriptionHandler(authManager),
		antigravity: newAntigravityTranscriptionHandler(authManager),
	}
}

func (h *transcriptionHandler) Handle(c *gin.Context) {
	if h == nil {
		writeTranscriptionError(c, http.StatusServiceUnavailable, "transcription handler unavailable", "transcription_unavailable")
		return
	}
	if c != nil && c.Request != nil && strings.HasPrefix(strings.ToLower(c.GetHeader("Content-Type")), "multipart/form-data;") {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, codexTranscriptionMaxRequest)
		_ = c.Request.ParseMultipartForm(32 << 20)
	}
	model := strings.ToLower(strings.TrimSpace(c.PostForm("model")))
	if isAntigravityTranscriptionModel(model) {
		h.antigravity.Handle(c)
		return
	}
	h.codex.Handle(c)
}

func isAntigravityTranscriptionModel(model string) bool {
	if _, ok := supportedAntigravityTranscriptionModels[model]; ok {
		return true
	}
	return strings.HasPrefix(model, "gemini-")
}

type antigravityTranscriptionHandler struct {
	authManager *auth.Manager
}

type antigravityTranscriptionRequest struct {
	clientModel    string
	upstreamModel  string
	responseFormat string
	payload        []byte
}

func newAntigravityTranscriptionHandler(authManager *auth.Manager) *antigravityTranscriptionHandler {
	return &antigravityTranscriptionHandler{authManager: authManager}
}

func (h *antigravityTranscriptionHandler) Handle(c *gin.Context) {
	if h == nil || h.authManager == nil {
		writeTranscriptionError(c, http.StatusServiceUnavailable, "Antigravity authentication manager unavailable", "antigravity_auth_unavailable")
		return
	}
	request, errParse := h.parseRequest(c)
	if errParse != nil {
		status := http.StatusBadRequest
		var tooLarge *transcriptionRequestTooLargeError
		if errors.As(errParse, &tooLarge) {
			status = http.StatusRequestEntityTooLarge
		}
		writeTranscriptionError(c, status, errParse.Error(), "invalid_request")
		return
	}
	if c.Request.MultipartForm != nil {
		defer func() { _ = c.Request.MultipartForm.RemoveAll() }()
	}

	response, errExecute := h.authManager.Execute(c.Request.Context(), []string{"antigravity"}, cliproxyexecutor.Request{
		Model:   request.upstreamModel,
		Payload: request.payload,
	}, cliproxyexecutor.Options{
		OriginalRequest: request.payload,
		SourceFormat:    sdktranslator.FormatGemini,
		ResponseFormat:  sdktranslator.FormatGemini,
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: request.clientModel,
			cliproxyexecutor.RequestPathMetadataKey:    "/v1/audio/transcriptions",
		},
	})
	if errExecute != nil {
		status, code, errorType := antigravityTranscriptionError(errExecute)
		writeTranscriptionError(c, status, "Google transcription request failed: "+safeTranscriptionError(errExecute), code, errorType)
		return
	}

	body, errRead := readAntigravityTranscriptionResponse(response.Payload)
	if errRead != nil {
		writeTranscriptionError(c, http.StatusBadGateway, errRead.Error(), "invalid_upstream_response")
		return
	}
	normalized, contentType, errNormalize := normalizeGoogleTranscriptionResponse(body, request.responseFormat)
	if errNormalize != nil {
		writeTranscriptionError(c, http.StatusBadGateway, errNormalize.Error(), "invalid_upstream_response")
		return
	}
	c.Data(http.StatusOK, contentType, normalized)
}

func (h *antigravityTranscriptionHandler) parseRequest(c *gin.Context) (antigravityTranscriptionRequest, error) {
	if c == nil || c.Request == nil {
		return antigravityTranscriptionRequest{}, errors.New("transcription request is missing")
	}
	if !strings.HasPrefix(strings.ToLower(c.GetHeader("Content-Type")), "multipart/form-data;") {
		return antigravityTranscriptionRequest{}, errors.New("Content-Type must be multipart/form-data")
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, codexTranscriptionMaxRequest)
	if errParse := c.Request.ParseMultipartForm(32 << 20); errParse != nil {
		if strings.Contains(strings.ToLower(errParse.Error()), "request body too large") {
			return antigravityTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
		}
		return antigravityTranscriptionRequest{}, fmt.Errorf("invalid multipart form: %w", errParse)
	}

	clientModel := strings.ToLower(strings.TrimSpace(c.PostForm("model")))
	upstreamModel, ok := supportedAntigravityTranscriptionModels[clientModel]
	if !ok {
		return antigravityTranscriptionRequest{}, fmt.Errorf("unsupported Google transcription model %q", clientModel)
	}
	responseFormat := strings.ToLower(strings.TrimSpace(c.PostForm("response_format")))
	if responseFormat == "" {
		responseFormat = "json"
	}
	switch responseFormat {
	case "json", "text", "verbose_json":
	case "srt", "vtt":
		return antigravityTranscriptionRequest{}, fmt.Errorf("response_format %q is unsupported because Antigravity does not provide timestamps", responseFormat)
	default:
		return antigravityTranscriptionRequest{}, fmt.Errorf("unsupported transcription response_format %q", responseFormat)
	}

	temperature := strings.TrimSpace(c.PostForm("temperature"))
	var temperatureValue *float64
	if temperature != "" {
		value, errParseTemperature := strconv.ParseFloat(temperature, 64)
		if errParseTemperature != nil || value < 0 || value > 1 {
			return antigravityTranscriptionRequest{}, errors.New("temperature must be a number between 0 and 1")
		}
		temperatureValue = &value
	}

	fileHeader, errFile := c.FormFile("file")
	if errFile != nil {
		return antigravityTranscriptionRequest{}, errors.New("file is required")
	}
	if fileHeader.Size <= 0 {
		return antigravityTranscriptionRequest{}, errors.New("file must not be empty")
	}
	if fileHeader.Size > codexTranscriptionMaxAudio {
		return antigravityTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
	}

	payload, errBuild := buildAntigravityTranscriptionPayload(
		fileHeader,
		strings.TrimSpace(c.PostForm("language")),
		strings.TrimSpace(c.PostForm("prompt")),
		temperatureValue,
	)
	if errBuild != nil {
		return antigravityTranscriptionRequest{}, errBuild
	}
	return antigravityTranscriptionRequest{
		clientModel:    clientModel,
		upstreamModel:  upstreamModel,
		responseFormat: responseFormat,
		payload:        payload,
	}, nil
}

func buildAntigravityTranscriptionPayload(fileHeader *multipart.FileHeader, language, prompt string, temperature *float64) ([]byte, error) {
	if fileHeader == nil {
		return nil, errors.New("file is required")
	}
	file, errOpen := fileHeader.Open()
	if errOpen != nil {
		return nil, fmt.Errorf("failed to open uploaded file: %w", errOpen)
	}
	defer func() { _ = file.Close() }()

	var encoded bytes.Buffer
	encoder := base64.NewEncoder(base64.StdEncoding, &encoded)
	written, errCopy := io.Copy(encoder, io.LimitReader(file, codexTranscriptionMaxAudio+1))
	if errClose := encoder.Close(); errCopy == nil && errClose != nil {
		errCopy = errClose
	}
	if errCopy != nil {
		return nil, fmt.Errorf("failed to read uploaded file: %w", errCopy)
	}
	if written > codexTranscriptionMaxAudio {
		return nil, &transcriptionRequestTooLargeError{}
	}

	mimeType := strings.TrimSpace(fileHeader.Header.Get("Content-Type"))
	if mimeType == "" || mimeType == "application/octet-stream" {
		mimeType = mime.TypeByExtension(filepath.Ext(fileHeader.Filename))
	}
	if strings.EqualFold(mimeType, "audio/x-wav") {
		mimeType = "audio/wav"
	}
	if mimeType == "" {
		mimeType = "audio/wav"
	}

	instruction := "Transcribe this audio verbatim. Return only the transcript, with no commentary."
	if language != "" {
		instruction += " The spoken language is " + language + "."
	}
	if prompt != "" {
		instruction += " Additional transcription context: " + prompt
	}

	generationConfig := map[string]any{}
	if temperature != nil {
		generationConfig["temperature"] = *temperature
	}
	payload := map[string]any{
		"contents": []any{
			map[string]any{
				"role": "user",
				"parts": []any{
					map[string]any{"text": instruction},
					map[string]any{"inline_data": map[string]string{
						"mime_type": mimeType,
						"data":      encoded.String(),
					}},
				},
			},
		},
	}
	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}
	return json.Marshal(payload)
}

func readAntigravityTranscriptionResponse(payload []byte) ([]byte, error) {
	if len(payload) > antigravityTranscriptionMaxResponse {
		return nil, errors.New("Google transcription response is too large")
	}
	if len(bytes.TrimSpace(payload)) == 0 {
		return nil, errors.New("Google transcription response was empty")
	}
	return payload, nil
}

func normalizeGoogleTranscriptionResponse(body []byte, responseFormat string) ([]byte, string, error) {
	text, errExtract := extractGeminiTranscriptionText(body)
	if errExtract != nil {
		return nil, "", errExtract
	}
	if responseFormat == "text" {
		return []byte(text), "text/plain; charset=utf-8", nil
	}
	normalized, errMarshal := json.Marshal(map[string]string{"text": text})
	if errMarshal != nil {
		return nil, "", fmt.Errorf("failed to encode transcription response: %w", errMarshal)
	}
	return normalized, "application/json; charset=utf-8", nil
}

func extractGeminiTranscriptionText(body []byte) (string, error) {
	var payload struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text    string `json:"text"`
					Thought bool   `json:"thought"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if errUnmarshal := json.Unmarshal(body, &payload); errUnmarshal != nil {
		return "", errors.New("Google transcription returned invalid JSON")
	}
	if message := strings.TrimSpace(payload.Error.Message); message != "" {
		return "", fmt.Errorf("Google transcription returned an error: %s", truncateTranscriptionError(message))
	}
	var text strings.Builder
	for _, candidate := range payload.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.Thought || part.Text == "" {
				continue
			}
			text.WriteString(part.Text)
		}
	}
	if len(payload.Candidates) == 0 || text.Len() == 0 {
		return "", errors.New("Google transcription response did not contain transcript text")
	}
	return text.String(), nil
}

func antigravityTranscriptionError(err error) (int, string, string) {
	status := http.StatusBadGateway
	code := "antigravity_upstream_error"
	errorType := "upstream_error"
	var statusError cliproxyexecutor.StatusError
	if errors.As(err, &statusError) {
		switch statusError.StatusCode() {
		case http.StatusUnauthorized, http.StatusForbidden:
			status = http.StatusServiceUnavailable
			code = "antigravity_auth_error"
			errorType = "authentication_error"
		case http.StatusTooManyRequests:
			status = http.StatusServiceUnavailable
			code = "antigravity_quota_error"
		}
	}
	var authError *auth.Error
	if errors.As(err, &authError) && (authError.Code == "auth_not_found" || authError.Code == "auth_unavailable") {
		status = http.StatusServiceUnavailable
		code = "antigravity_auth_unavailable"
		errorType = "authentication_error"
	}
	return status, code, errorType
}
