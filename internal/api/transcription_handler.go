package api

import (
	"bytes"
	"context"
	"encoding/json"
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
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
)

const (
	codexTranscriptionUpstreamURL = "https://chatgpt.com/backend-api/transcribe"
	codexTranscriptionMaxAudio    = 50 << 20
	codexTranscriptionMaxRequest  = codexTranscriptionMaxAudio + 1<<20
	codexTranscriptionMaxResponse = 8 << 20
	codexTranscriptionUserAgent   = "codex-tui/0.146.0 (Mac OS 26.5.0; arm64) iTerm.app/3.6.10 (codex-tui; 0.146.0)"
	codexTranscriptionOriginator  = "codex-tui"
)

var supportedCodexTranscriptionModels = map[string]struct{}{
	"whisper-1":                         {},
	"gpt-transcribe":                    {},
	"gpt-4o-transcribe":                 {},
	"gpt-4o-mini-transcribe":            {},
	"gpt-4o-transcribe-2025-12-15":      {},
	"gpt-4o-mini-transcribe-2025-12-15": {},
}

type codexTranscriptionHandler struct {
	authManager *auth.Manager
	upstreamURL string
}

type codexTranscriptionRequest struct {
	model          string
	language       string
	responseFormat string
	upstreamBody   []byte
	contentType    string
}

func newCodexTranscriptionHandler(authManager *auth.Manager) *codexTranscriptionHandler {
	return &codexTranscriptionHandler{
		authManager: authManager,
		upstreamURL: codexTranscriptionUpstreamURL,
	}
}

func (h *codexTranscriptionHandler) setUpstreamURL(upstreamURL string) {
	if h == nil || strings.TrimSpace(upstreamURL) == "" {
		return
	}
	h.upstreamURL = strings.TrimRight(strings.TrimSpace(upstreamURL), "/")
}

// Handle accepts an OpenAI-compatible multipart transcription request and
// forwards it to the authenticated Codex batch transcription endpoint.
func (h *codexTranscriptionHandler) Handle(c *gin.Context) {
	if h == nil || h.authManager == nil {
		writeTranscriptionError(c, http.StatusServiceUnavailable, "Codex authentication manager unavailable", "codex_auth_unavailable")
		return
	}

	request, errParse := h.parseRequest(c)
	if errParse != nil {
		status := transcriptionRequestErrorStatus(errParse)
		writeTranscriptionError(c, status, errParse.Error(), "invalid_request")
		return
	}
	if c.Request.MultipartForm != nil {
		defer func() { _ = c.Request.MultipartForm.RemoveAll() }()
	}

	selected, errSelect := h.authManager.SelectAuthByKind(
		c.Request.Context(),
		"codex",
		"",
		auth.AuthKindOAuth,
		cliproxyexecutor.Options{Metadata: map[string]any{
			cliproxyexecutor.RequestedModelMetadataKey: request.model,
		}},
	)
	if errSelect != nil || selected == nil {
		message := "Codex OAuth credential unavailable"
		if errSelect != nil {
			message += ": " + safeTranscriptionError(errSelect)
		}
		writeTranscriptionError(c, http.StatusServiceUnavailable, message, "codex_auth_unavailable")
		return
	}

	accounting := newCodexTranscriptionAccounting(c.Request.Context(), h.authManager, request.model, selected)

	response, refreshedAuth, errUpstream := h.doRequestWithRefresh(c.Request.Context(), selected, request)
	accounting.setAuth(refreshedAuth)
	if errUpstream != nil {
		accounting.finish(http.StatusBadGateway, errUpstream)
		writeTranscriptionError(c, http.StatusBadGateway, "Codex transcription upstream request failed: "+safeTranscriptionError(errUpstream), "upstream_error")
		return
	}
	if response == nil {
		errResponse := errors.New("Codex transcription upstream returned no response")
		accounting.finish(http.StatusBadGateway, errResponse)
		writeTranscriptionError(c, http.StatusBadGateway, errResponse.Error(), "upstream_error")
		return
	}
	if response.Body == nil {
		errResponse := errors.New("Codex transcription upstream returned an empty response body")
		accounting.finish(http.StatusBadGateway, errResponse)
		writeTranscriptionError(c, http.StatusBadGateway, errResponse.Error(), "upstream_error")
		return
	}
	defer func() { _ = response.Body.Close() }()
	accounting.observeResponse(response)

	body, errRead := io.ReadAll(io.LimitReader(response.Body, codexTranscriptionMaxResponse+1))
	if errRead != nil {
		errResponse := errors.New("failed to read Codex transcription response")
		accounting.finish(http.StatusBadGateway, errResponse)
		writeTranscriptionError(c, http.StatusBadGateway, "failed to read Codex transcription response", "upstream_error")
		return
	}
	if len(body) > codexTranscriptionMaxResponse {
		errResponse := errors.New("Codex transcription response is too large")
		accounting.finish(http.StatusBadGateway, errResponse)
		writeTranscriptionError(c, http.StatusBadGateway, "Codex transcription response is too large", "upstream_error")
		return
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		upstreamErr := newCodexTranscriptionHTTPError(response.StatusCode, body)
		accounting.finish(response.StatusCode, upstreamErr)
		handleTranscriptionUpstreamError(c, response.StatusCode, body)
		return
	}

	normalizedBody, contentType, errNormalize := normalizeTranscriptionResponse(body, request.responseFormat)
	if errNormalize != nil {
		accounting.finish(http.StatusBadGateway, errNormalize)
		writeTranscriptionError(c, http.StatusBadGateway, errNormalize.Error(), "invalid_upstream_response")
		return
	}
	accounting.finish(response.StatusCode, nil)
	c.Data(http.StatusOK, contentType, normalizedBody)
}

func (h *codexTranscriptionHandler) parseRequest(c *gin.Context) (codexTranscriptionRequest, error) {
	if c == nil || c.Request == nil {
		return codexTranscriptionRequest{}, errors.New("transcription request is missing")
	}
	if errParse := parseTranscriptionMultipartForm(c); errParse != nil {
		if strings.Contains(strings.ToLower(errParse.Error()), "request body too large") {
			return codexTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
		}
		return codexTranscriptionRequest{}, fmt.Errorf("invalid multipart form: %w", errParse)
	}

	model := strings.ToLower(strings.TrimSpace(c.PostForm("model")))
	if model == "" {
		return codexTranscriptionRequest{}, errors.New("model is required")
	}
	if _, ok := supportedCodexTranscriptionModels[model]; !ok {
		return codexTranscriptionRequest{}, fmt.Errorf("unsupported transcription model %q", model)
	}

	responseFormat := strings.ToLower(strings.TrimSpace(c.PostForm("response_format")))
	if responseFormat == "" {
		responseFormat = "json"
	}
	switch responseFormat {
	case "json", "text", "verbose_json":
	case "srt", "vtt":
		return codexTranscriptionRequest{}, fmt.Errorf("response_format %q is unsupported because the Codex transcription endpoint does not provide timestamps", responseFormat)
	default:
		return codexTranscriptionRequest{}, fmt.Errorf("unsupported transcription response_format %q", responseFormat)
	}
	if temperature := strings.TrimSpace(c.PostForm("temperature")); temperature != "" {
		value, errParseTemperature := strconv.ParseFloat(temperature, 64)
		if errParseTemperature != nil || value < 0 || value > 1 {
			return codexTranscriptionRequest{}, errors.New("temperature must be a number between 0 and 1")
		}
	}

	fileHeader, errFile := c.FormFile("file")
	if errFile != nil {
		return codexTranscriptionRequest{}, errors.New("file is required")
	}
	if fileHeader.Size <= 0 {
		return codexTranscriptionRequest{}, errors.New("file must not be empty")
	}
	if fileHeader.Size > codexTranscriptionMaxAudio {
		return codexTranscriptionRequest{}, &transcriptionRequestTooLargeError{}
	}

	upstreamBody, contentType, errBuild := buildCodexTranscriptionMultipart(fileHeader, strings.TrimSpace(c.PostForm("language")))
	if errBuild != nil {
		return codexTranscriptionRequest{}, errBuild
	}
	return codexTranscriptionRequest{
		model:          model,
		language:       strings.TrimSpace(c.PostForm("language")),
		responseFormat: responseFormat,
		upstreamBody:   upstreamBody,
		contentType:    contentType,
	}, nil
}

func buildCodexTranscriptionMultipart(fileHeader *multipart.FileHeader, language string) ([]byte, string, error) {
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
	contentType := strings.TrimSpace(fileHeader.Header.Get("Content-Type"))
	if contentType == "" {
		contentType = mime.TypeByExtension(filepath.Ext(filename))
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	filePartHeader := make(textproto.MIMEHeader)
	filePartHeader.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	filePartHeader.Set("Content-Type", contentType)
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
	if language != "" {
		if errField := writer.WriteField("language", language); errField != nil {
			return nil, "", fmt.Errorf("failed to write upstream language field: %w", errField)
		}
	}
	if errClose := writer.Close(); errClose != nil {
		return nil, "", fmt.Errorf("failed to close upstream multipart form: %w", errClose)
	}
	return body.Bytes(), writer.FormDataContentType(), nil
}

func (h *codexTranscriptionHandler) doRequestWithRefresh(ctx context.Context, selected *auth.Auth, request codexTranscriptionRequest) (*http.Response, *auth.Auth, error) {
	response, errRequest := h.doRequest(ctx, selected, request)
	if errRequest != nil || response == nil || response.StatusCode != http.StatusUnauthorized || !hasCodexRefreshToken(selected) {
		return response, selected, errRequest
	}

	failedAccessToken := codexAuthMetadataString(selected, "access_token", "accessToken")
	originalBody, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	_ = response.Body.Close()
	response.Body = io.NopCloser(bytes.NewReader(originalBody))
	refreshed, errRefresh := h.authManager.RefreshAuthForRequest(ctx, selected.ID, failedAccessToken)
	if errRefresh != nil || refreshed == nil {
		return response, selected, nil
	}
	refreshedResponse, errRetry := h.doRequest(ctx, refreshed, request)
	return refreshedResponse, refreshed, errRetry
}

func (h *codexTranscriptionHandler) doRequest(ctx context.Context, selected *auth.Auth, request codexTranscriptionRequest) (*http.Response, error) {
	headers := make(http.Header)
	headers.Set("Content-Type", request.contentType)
	headers.Set("Accept", "application/json")
	headers.Set("Originator", codexTranscriptionOriginator)
	headers.Set("User-Agent", codexTranscriptionUserAgent)
	if accountID := codexAuthMetadataString(selected, "account_id", "chatgpt_account_id"); accountID != "" {
		headers.Set("Chatgpt-Account-Id", accountID)
	}
	upstreamURL := h.upstreamURL
	if strings.TrimSpace(upstreamURL) == "" {
		upstreamURL = codexTranscriptionUpstreamURL
	}
	httpRequest, errNewRequest := h.authManager.NewHttpRequest(ctx, selected, http.MethodPost, upstreamURL, request.upstreamBody, headers)
	if errNewRequest != nil {
		return nil, fmt.Errorf("failed to prepare Codex transcription request: %w", errNewRequest)
	}
	return h.authManager.HttpRequest(ctx, selected, httpRequest)
}

func normalizeTranscriptionResponse(body []byte, responseFormat string) ([]byte, string, error) {
	text, errExtract := extractTranscriptionText(body)
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

func extractTranscriptionText(body []byte) (string, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return "", nil
	}
	if trimmed[0] == '{' {
		var payload struct {
			Text *string `json:"text"`
		}
		if errUnmarshal := json.Unmarshal(trimmed, &payload); errUnmarshal != nil {
			return "", errors.New("Codex transcription returned invalid JSON")
		}
		if payload.Text == nil {
			return "", errors.New("Codex transcription response did not contain a text field")
		}
		return *payload.Text, nil
	}
	if trimmed[0] == '"' {
		var text string
		if errUnmarshal := json.Unmarshal(trimmed, &text); errUnmarshal == nil {
			return text, nil
		}
	}
	return string(trimmed), nil
}

func handleTranscriptionUpstreamError(c *gin.Context, status int, body []byte) {
	message := "Codex transcription upstream returned HTTP " + strconv.Itoa(status)
	if detail := transcriptionUpstreamErrorDetail(body); detail != "" {
		message += ": " + detail
	}
	errorCode := "codex_transcription_upstream_error"
	errorType := "upstream_error"
	if status == http.StatusUnauthorized || status == http.StatusForbidden {
		errorCode = "codex_auth_error"
		errorType = "authentication_error"
	}
	if status >= http.StatusInternalServerError {
		status = http.StatusBadGateway
	}
	writeTranscriptionError(c, status, message, errorCode, errorType)
}

func transcriptionUpstreamErrorDetail(body []byte) string {
	var payload struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &payload) == nil {
		if message := strings.TrimSpace(payload.Error.Message); message != "" {
			return truncateTranscriptionError(message)
		}
		if message := strings.TrimSpace(payload.Message); message != "" {
			return truncateTranscriptionError(message)
		}
	}
	return truncateTranscriptionError(string(bytes.TrimSpace(body)))
}

func truncateTranscriptionError(message string) string {
	message = strings.Join(strings.Fields(message), " ")
	if len(message) > 512 {
		message = message[:512] + "..."
	}
	if strings.HasPrefix(strings.ToLower(message), "<!doctype") || strings.HasPrefix(strings.ToLower(message), "<html") {
		return "upstream returned an HTML error page"
	}
	return message
}

func writeTranscriptionError(c *gin.Context, status int, message, code string, errorType ...string) {
	typeValue := "invalid_request_error"
	if len(errorType) > 0 && strings.TrimSpace(errorType[0]) != "" {
		typeValue = errorType[0]
	}
	c.JSON(status, gin.H{"error": gin.H{
		"message": message,
		"type":    typeValue,
		"code":    code,
	}})
}

func hasCodexRefreshToken(selected *auth.Auth) bool {
	return codexAuthMetadataString(selected, "refresh_token", "refreshToken") != ""
}

func codexAuthMetadataString(selected *auth.Auth, keys ...string) string {
	if selected == nil || selected.Metadata == nil {
		return ""
	}
	for _, key := range keys {
		if value, ok := selected.Metadata[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func safeTranscriptionError(err error) string {
	if err == nil {
		return "unknown error"
	}
	return truncateTranscriptionError(err.Error())
}

type transcriptionRequestTooLargeError struct{}

func (*transcriptionRequestTooLargeError) Error() string {
	return "audio upload exceeds the 50 MiB limit"
}

type codexTranscriptionHTTPError struct {
	status  int
	message string
}

func (e *codexTranscriptionHTTPError) Error() string {
	if e == nil {
		return "Codex transcription upstream request failed"
	}
	return e.message
}

func (e *codexTranscriptionHTTPError) StatusCode() int {
	if e == nil {
		return 0
	}
	return e.status
}

func transcriptionRequestErrorStatus(err error) int {
	var tooLarge *transcriptionRequestTooLargeError
	if errors.As(err, &tooLarge) || strings.Contains(strings.ToLower(safeTranscriptionError(err)), "request body too large") {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

func newCodexTranscriptionHTTPError(status int, body []byte) error {
	message := "Codex transcription upstream returned HTTP " + strconv.Itoa(status)
	if detail := transcriptionUpstreamErrorDetail(body); detail != "" {
		message += ": " + detail
	}
	return &codexTranscriptionHTTPError{status: status, message: message}
}

func codexTranscriptionResultError(err error, status int) *auth.Error {
	if err == nil {
		err = errors.New("Codex transcription request failed")
	}
	if status <= 0 {
		status = http.StatusBadGateway
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return auth.NewRequestScopedError(safeTranscriptionError(err), status)
	}
	var authErr *auth.Error
	if errors.As(err, &authErr) && authErr != nil {
		return authErr
	}
	return &auth.Error{
		Code:       "codex_transcription_error",
		Message:    safeTranscriptionError(err),
		HTTPStatus: status,
	}
}

type codexTranscriptionAccounting struct {
	ctx       context.Context
	manager   *auth.Manager
	model     string
	auth      *auth.Auth
	reporter  *helps.UsageReporter
	completed bool
}

func newCodexTranscriptionAccounting(ctx context.Context, manager *auth.Manager, model string, selected *auth.Auth) *codexTranscriptionAccounting {
	return &codexTranscriptionAccounting{
		ctx:      ctx,
		manager:  manager,
		model:    model,
		auth:     selected,
		reporter: helps.NewUsageReporter(ctx, "codex", model, selected),
	}
}

func (a *codexTranscriptionAccounting) setAuth(selected *auth.Auth) {
	if a == nil || selected == nil {
		return
	}
	a.auth = selected
	a.reporter.UpdateAccessTokenFingerprint(selected)
}

func (a *codexTranscriptionAccounting) observeResponse(response *http.Response) {
	if a == nil {
		return
	}
	a.reporter.ObserveResponse(response)
}

func (a *codexTranscriptionAccounting) finish(status int, err error) {
	if a == nil || a.completed {
		return
	}
	a.completed = true
	success := err == nil
	if a.manager != nil && a.auth != nil {
		result := auth.Result{
			AuthID:   a.auth.ID,
			Provider: "codex",
			Model:    a.model,
			Success:  success,
			Options: cliproxyexecutor.Options{Metadata: map[string]any{
				cliproxyexecutor.RequestedModelMetadataKey: a.model,
			}},
		}
		if !success {
			result.Error = codexTranscriptionResultError(err, status)
		}
		a.manager.MarkResult(a.ctx, result)
	}
	if success {
		a.reporter.EnsurePublished(a.ctx)
	} else {
		a.reporter.PublishFailure(a.ctx, err)
	}
}
