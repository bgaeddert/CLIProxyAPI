package api

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

const openAICompatTranscriptionHandlerType = "openai-transcription"

type openAICompatTranscriptionHandler struct {
	authManager *auth.Manager
}

type openAICompatTranscriptionRequest struct {
	model          string
	responseFormat string
	upstreamBody   []byte
	contentType    string
}

func newOpenAICompatTranscriptionHandler(authManager *auth.Manager) *openAICompatTranscriptionHandler {
	return &openAICompatTranscriptionHandler{authManager: authManager}
}

func (h *openAICompatTranscriptionHandler) Handle(c *gin.Context) {
	if h == nil || h.authManager == nil {
		writeTranscriptionError(c, http.StatusServiceUnavailable, "OpenAI-compatible authentication manager unavailable", "openai_compat_auth_unavailable")
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

	providers := util.GetProviderName(request.model)
	if len(providers) == 0 {
		writeTranscriptionError(c, http.StatusServiceUnavailable, "no OpenAI-compatible provider is available for transcription model "+strconv.Quote(request.model), "provider_unavailable", "server_error")
		return
	}

	response, errExecute := h.authManager.Execute(c.Request.Context(), providers, cliproxyexecutor.Request{
		Model:   request.model,
		Payload: request.upstreamBody,
	}, cliproxyexecutor.Options{
		OriginalRequest: request.upstreamBody,
		SourceFormat:    sdktranslator.FromString(openAICompatTranscriptionHandlerType),
		ResponseFormat:  sdktranslator.FromString(openAICompatTranscriptionHandlerType),
		Headers: http.Header{
			"Content-Type": []string{request.contentType},
		},
		Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: request.model,
			cliproxyexecutor.RequestPathMetadataKey:    "/v1/audio/transcriptions",
		},
	})
	if errExecute != nil {
		status, code, errorType := openAICompatTranscriptionError(errExecute)
		writeTranscriptionError(c, status, "OpenAI-compatible transcription request failed: "+safeTranscriptionError(errExecute), code, errorType)
		return
	}
	if len(response.Payload) > codexTranscriptionMaxResponse {
		writeTranscriptionError(c, http.StatusBadGateway, "OpenAI-compatible transcription response is too large", "invalid_upstream_response")
		return
	}

	normalized, contentType, errNormalize := normalizeTranscriptionResponse(response.Payload, request.responseFormat)
	if errNormalize != nil {
		writeTranscriptionError(c, http.StatusBadGateway, errNormalize.Error(), "invalid_upstream_response")
		return
	}
	c.Data(http.StatusOK, contentType, normalized)
}

func (h *openAICompatTranscriptionHandler) parseRequest(c *gin.Context) (openAICompatTranscriptionRequest, error) {
	if c == nil || c.Request == nil {
		return openAICompatTranscriptionRequest{}, errors.New("transcription request is missing")
	}
	if !strings.HasPrefix(strings.ToLower(c.GetHeader("Content-Type")), "multipart/form-data;") {
		return openAICompatTranscriptionRequest{}, errors.New("Content-Type must be multipart/form-data")
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, codexTranscriptionMaxRequest)
	if errParse := c.Request.ParseMultipartForm(32 << 20); errParse != nil {
		if strings.Contains(strings.ToLower(errParse.Error()), "request body too large") {
			return openAICompatTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
		}
		return openAICompatTranscriptionRequest{}, fmt.Errorf("invalid multipart form: %w", errParse)
	}

	model := strings.ToLower(strings.TrimSpace(c.PostForm("model")))
	if model == "" {
		return openAICompatTranscriptionRequest{}, errors.New("model is required")
	}
	info := registry.LookupModelInfo(model)
	if !registry.ModelSupportsTranscriptionEndpoints(info) {
		return openAICompatTranscriptionRequest{}, fmt.Errorf("model %q is not configured for transcription endpoints", model)
	}

	responseFormat := strings.ToLower(strings.TrimSpace(c.PostForm("response_format")))
	if responseFormat == "" {
		responseFormat = "json"
	}
	switch responseFormat {
	case "json", "text", "verbose_json":
	case "srt", "vtt":
		return openAICompatTranscriptionRequest{}, fmt.Errorf("response_format %q is unsupported because this transcription path does not provide timestamps", responseFormat)
	default:
		return openAICompatTranscriptionRequest{}, fmt.Errorf("unsupported transcription response_format %q", responseFormat)
	}

	temperature := strings.TrimSpace(c.PostForm("temperature"))
	if temperature != "" {
		value, errParseTemperature := strconv.ParseFloat(temperature, 64)
		if errParseTemperature != nil || value < 0 || value > 1 {
			return openAICompatTranscriptionRequest{}, errors.New("temperature must be a number between 0 and 1")
		}
	}

	fileHeader, errFile := c.FormFile("file")
	if errFile != nil {
		return openAICompatTranscriptionRequest{}, errors.New("file is required")
	}
	if fileHeader.Size <= 0 {
		return openAICompatTranscriptionRequest{}, errors.New("file must not be empty")
	}
	if fileHeader.Size > codexTranscriptionMaxAudio {
		return openAICompatTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
	}

	upstreamBody, contentType, errBuild := buildOpenAICompatTranscriptionMultipart(
		fileHeader,
		model,
		strings.TrimSpace(c.PostForm("language")),
		c.PostForm("prompt"),
		responseFormat,
		temperature,
	)
	if errBuild != nil {
		return openAICompatTranscriptionRequest{}, errBuild
	}
	return openAICompatTranscriptionRequest{
		model:          model,
		responseFormat: responseFormat,
		upstreamBody:   upstreamBody,
		contentType:    contentType,
	}, nil
}

func buildOpenAICompatTranscriptionMultipart(fileHeader *multipart.FileHeader, model, language, prompt, responseFormat, temperature string) ([]byte, string, error) {
	if fileHeader == nil {
		return nil, "", errors.New("file is required")
	}
	file, errOpen := fileHeader.Open()
	if errOpen != nil {
		return nil, "", fmt.Errorf("failed to open uploaded file: %w", errOpen)
	}
	defer func() { _ = file.Close() }()

	filename := filepath.Base(strings.TrimSpace(fileHeader.Filename))
	if filename == "." || filename == "" || filename == string(filepath.Separator) {
		filename = "audio"
	}
	filename = strings.NewReplacer(`\`, "_", `"`, "_").Replace(filename)
	fileContentType := strings.TrimSpace(fileHeader.Header.Get("Content-Type"))
	if fileContentType == "" {
		fileContentType = mime.TypeByExtension(filepath.Ext(filename))
	}
	if fileContentType == "" {
		fileContentType = "application/octet-stream"
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	filePartHeader := make(textproto.MIMEHeader)
	filePartHeader.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	filePartHeader.Set("Content-Type", fileContentType)
	part, errPart := writer.CreatePart(filePartHeader)
	if errPart != nil {
		return nil, "", fmt.Errorf("failed to create upstream file part: %w", errPart)
	}
	written, errCopy := io.Copy(part, io.LimitReader(file, codexTranscriptionMaxAudio+1))
	if errCopy != nil {
		return nil, "", fmt.Errorf("failed to read uploaded file: %w", errCopy)
	}
	if written > codexTranscriptionMaxAudio {
		return nil, "", &transcriptionRequestTooLargeError{}
	}
	if model != "" {
		if errField := writer.WriteField("model", model); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream model field: %w", errField)
		}
	}
	if language != "" {
		if errField := writer.WriteField("language", language); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream language field: %w", errField)
		}
	}
	if strings.TrimSpace(prompt) != "" {
		if errField := writer.WriteField("prompt", prompt); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream prompt field: %w", errField)
		}
	}
	if responseFormat != "" {
		if errField := writer.WriteField("response_format", responseFormat); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream response_format field: %w", errField)
		}
	}
	if temperature != "" {
		if errField := writer.WriteField("temperature", temperature); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream temperature field: %w", errField)
		}
	}
	if errClose := writer.Close(); errClose != nil {
		return nil, "", fmt.Errorf("failed to close upstream multipart form: %w", errClose)
	}
	return body.Bytes(), writer.FormDataContentType(), nil
}

func openAICompatTranscriptionError(err error) (int, string, string) {
	status := http.StatusBadGateway
	code := "openai_compat_upstream_error"
	errorType := "upstream_error"
	var statusError cliproxyexecutor.StatusError
	if errors.As(err, &statusError) {
		switch statusError.StatusCode() {
		case http.StatusUnauthorized, http.StatusForbidden:
			status = http.StatusServiceUnavailable
			code = "openai_compat_auth_error"
			errorType = "authentication_error"
		case http.StatusTooManyRequests:
			status = http.StatusServiceUnavailable
			code = "openai_compat_quota_error"
		}
	}
	var authError *auth.Error
	if errors.As(err, &authError) && (authError.Code == "auth_not_found" || authError.Code == "auth_unavailable" || authError.Code == "provider_not_found") {
		status = http.StatusServiceUnavailable
		code = "openai_compat_auth_unavailable"
		errorType = "authentication_error"
	}
	return status, code, errorType
}
