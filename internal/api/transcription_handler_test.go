package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
)

type transcriptionTestExecutor struct {
	refreshCalls atomic.Int32
	httpCalls    atomic.Int32
}

type antigravityTranscriptionTestExecutor struct {
	responsePayload []byte
	executeErr      error
	request         coreexecutor.Request
	options         coreexecutor.Options
	selectedAuth    *auth.Auth
	calls           atomic.Int32
}

type transcriptionUsagePluginFunc func(context.Context, usage.Record)

func (f transcriptionUsagePluginFunc) HandleUsage(ctx context.Context, record usage.Record) {
	f(ctx, record)
}

func (*antigravityTranscriptionTestExecutor) Identifier() string { return "antigravity" }

func (e *antigravityTranscriptionTestExecutor) Execute(_ context.Context, selected *auth.Auth, request coreexecutor.Request, options coreexecutor.Options) (coreexecutor.Response, error) {
	e.calls.Add(1)
	e.selectedAuth = selected.Clone()
	e.request = request
	e.options = options
	if e.executeErr != nil {
		return coreexecutor.Response{}, e.executeErr
	}
	return coreexecutor.Response{Payload: append([]byte(nil), e.responsePayload...)}, nil
}

func (*antigravityTranscriptionTestExecutor) ExecuteStream(context.Context, *auth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	return nil, nil
}

func (*antigravityTranscriptionTestExecutor) Refresh(_ context.Context, selected *auth.Auth) (*auth.Auth, error) {
	return selected.Clone(), nil
}

func (*antigravityTranscriptionTestExecutor) CountTokens(context.Context, *auth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, nil
}

func (*antigravityTranscriptionTestExecutor) PrepareRequest(*http.Request, *auth.Auth) error {
	return nil
}

func (*antigravityTranscriptionTestExecutor) HttpRequest(context.Context, *auth.Auth, *http.Request) (*http.Response, error) {
	return nil, nil
}

type transcriptionStatusError struct {
	status  int
	message string
}

func (e transcriptionStatusError) Error() string { return e.message }

func (e transcriptionStatusError) StatusCode() int { return e.status }

func (*transcriptionTestExecutor) Identifier() string { return "codex" }

func (*transcriptionTestExecutor) Execute(context.Context, *auth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, nil
}

func (*transcriptionTestExecutor) ExecuteStream(context.Context, *auth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	return nil, nil
}

func (e *transcriptionTestExecutor) Refresh(_ context.Context, selected *auth.Auth) (*auth.Auth, error) {
	e.refreshCalls.Add(1)
	refreshed := selected.Clone()
	if refreshed.Metadata == nil {
		refreshed.Metadata = make(map[string]any)
	}
	refreshed.Metadata["access_token"] = "refreshed-token"
	return refreshed, nil
}

func (*transcriptionTestExecutor) CountTokens(context.Context, *auth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, nil
}

func (*transcriptionTestExecutor) PrepareRequest(req *http.Request, selected *auth.Auth) error {
	if selected != nil && selected.Metadata != nil {
		if token, ok := selected.Metadata["access_token"].(string); ok {
			req.Header.Set("Authorization", "Bearer "+token)
		}
	}
	return nil
}

func (e *transcriptionTestExecutor) HttpRequest(ctx context.Context, _ *auth.Auth, req *http.Request) (*http.Response, error) {
	e.httpCalls.Add(1)
	return http.DefaultClient.Do(req.WithContext(ctx))
}

func newTranscriptionTestHandler(t *testing.T, upstream http.HandlerFunc, withRefreshToken bool) (*codexTranscriptionHandler, *transcriptionTestExecutor, func()) {
	t.Helper()
	upstreamServer := httptest.NewServer(upstream)
	manager := auth.NewManager(nil, nil, nil)
	executor := &transcriptionTestExecutor{}
	manager.RegisterExecutor(executor)
	metadata := map[string]any{
		"access_token": "codex-token",
		"account_id":   "account-123",
	}
	if withRefreshToken {
		metadata["refresh_token"] = "refresh-token"
	}
	if _, errRegister := manager.Register(context.Background(), &auth.Auth{
		ID:       "codex-oauth",
		Provider: "codex",
		Status:   auth.StatusActive,
		Metadata: metadata,
	}); errRegister != nil {
		t.Fatalf("register Codex auth: %v", errRegister)
	}
	handler := newCodexTranscriptionHandler(manager)
	handler.setUpstreamURL(upstreamServer.URL + "/backend-api/transcribe")
	return handler, executor, upstreamServer.Close
}

func transcriptionRequest(t *testing.T, fields map[string]string, includeFile bool) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range fields {
		if errField := writer.WriteField(key, value); errField != nil {
			t.Fatalf("write form field %q: %v", key, errField)
		}
	}
	if includeFile {
		part, errFile := writer.CreateFormFile("file", "sample.wav")
		if errFile != nil {
			t.Fatalf("create form file: %v", errFile)
		}
		if _, errWrite := part.Write([]byte("RIFF-test-audio")); errWrite != nil {
			t.Fatalf("write form file: %v", errWrite)
		}
	}
	if errClose := writer.Close(); errClose != nil {
		t.Fatalf("close multipart writer: %v", errClose)
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	return request
}

func TestParseTranscriptionMultipartFormIsIdempotent(t *testing.T) {
	request := transcriptionRequest(t, map[string]string{"model": "whisper-1"}, true)
	recorder := httptest.NewRecorder()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(recorder)
	c.Request = request

	if errParse := parseTranscriptionMultipartForm(c); errParse != nil {
		t.Fatalf("first multipart parse: %v", errParse)
	}
	parsedForm := c.Request.MultipartForm
	if parsedForm == nil {
		t.Fatal("multipart form was not parsed")
	}
	if errParse := parseTranscriptionMultipartForm(c); errParse != nil {
		t.Fatalf("second multipart parse: %v", errParse)
	}
	if c.Request.MultipartForm != parsedForm {
		t.Fatal("second multipart parse replaced the parsed form")
	}
}

func invokeTranscriptionHandler(t *testing.T, handler *codexTranscriptionHandler, request *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(recorder)
	context.Request = request
	handler.Handle(context)
	return recorder
}

func invokeAntigravityTranscriptionHandler(t *testing.T, handler *antigravityTranscriptionHandler, request *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(recorder)
	context.Request = request
	handler.Handle(context)
	return recorder
}

func newAntigravityTranscriptionTestHandler(t *testing.T, responsePayload []byte, executeErr error) (*antigravityTranscriptionHandler, *antigravityTranscriptionTestExecutor) {
	t.Helper()
	registryRef := registry.GetGlobalRegistry()
	registryRef.RegisterClient("antigravity-oauth", "antigravity", []*registry.ModelInfo{{ID: antigravityTranscriptionDefaultModel}})
	t.Cleanup(func() { registryRef.UnregisterClient("antigravity-oauth") })
	manager := auth.NewManager(nil, nil, nil)
	executor := &antigravityTranscriptionTestExecutor{
		responsePayload: responsePayload,
		executeErr:      executeErr,
	}
	manager.RegisterExecutor(executor)
	if _, errRegister := manager.Register(context.Background(), &auth.Auth{
		ID:       "antigravity-oauth",
		Provider: "antigravity",
		Status:   auth.StatusActive,
		Metadata: map[string]any{
			"access_token": "antigravity-token",
			"project_id":   "project-123",
		},
	}); errRegister != nil {
		t.Fatalf("register Antigravity auth: %v", errRegister)
	}
	return newAntigravityTranscriptionHandler(manager), executor
}

func TestCodexTranscriptionForwardsMultipartAndNormalizesJSON(t *testing.T) {
	handler, _, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/backend-api/transcribe" {
			t.Errorf("upstream request = %s %s, want POST /backend-api/transcribe", request.Method, request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer codex-token" {
			t.Errorf("Authorization = %q, want OAuth bearer", got)
		}
		if got := request.Header.Get("Chatgpt-Account-Id"); got != "account-123" {
			t.Errorf("Chatgpt-Account-Id = %q, want account-123", got)
		}
		if got := request.Header.Get("Originator"); got != codexTranscriptionOriginator {
			t.Errorf("Originator = %q, want %q", got, codexTranscriptionOriginator)
		}
		if got := request.Header.Get("User-Agent"); got != codexTranscriptionUserAgent {
			t.Errorf("User-Agent = %q, want Codex user agent", got)
		}
		if errParse := request.ParseMultipartForm(1 << 20); errParse != nil {
			t.Errorf("parse upstream multipart form: %v", errParse)
			return
		}
		if got := request.FormValue("language"); got != "en" {
			t.Errorf("language = %q, want en", got)
		}
		if got := request.FormValue("model"); got != "" {
			t.Errorf("model = %q, want private upstream model field omitted", got)
		}
		fileHeaders := request.MultipartForm.File["file"]
		if len(fileHeaders) != 1 {
			t.Errorf("upstream file count = %d, want 1", len(fileHeaders))
			return
		}
		file, errOpen := fileHeaders[0].Open()
		if errOpen != nil {
			t.Errorf("open upstream file: %v", errOpen)
			return
		}
		defer func() { _ = file.Close() }()
		fileBody, errRead := io.ReadAll(file)
		if errRead != nil || string(fileBody) != "RIFF-test-audio" {
			t.Errorf("upstream file body = %q, read error = %v", fileBody, errRead)
		}
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"hello from Codex"}`))
	}, false)
	defer closeServer()

	recorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":    "whisper-1",
		"language": "en",
	}, true))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var response struct {
		Text string `json:"text"`
	}
	if errUnmarshal := json.Unmarshal(recorder.Body.Bytes(), &response); errUnmarshal != nil {
		t.Fatalf("decode response: %v", errUnmarshal)
	}
	if response.Text != "hello from Codex" {
		t.Fatalf("text = %q, want hello from Codex", response.Text)
	}
	gotAuth, ok := handler.authManager.GetByID("codex-oauth")
	if !ok || gotAuth == nil {
		t.Fatal("Codex auth was not retained by the manager")
	}
	if gotAuth.Success != 1 || gotAuth.Failed != 0 {
		t.Fatalf("auth totals = success=%d failed=%d, want 1/0", gotAuth.Success, gotAuth.Failed)
	}
	var recentSuccess int64
	for _, bucket := range gotAuth.RecentRequestsSnapshot(time.Now()) {
		recentSuccess += bucket.Success
	}
	if recentSuccess != 1 {
		t.Fatalf("recent successful requests = %d, want 1", recentSuccess)
	}
}

func TestCodexTranscriptionPublishesUsageRecord(t *testing.T) {
	handler, _, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, _ *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"usage test"}`))
	}, false)
	defer closeServer()

	records := make(chan usage.Record, 1)
	pluginName := "transcription-test-usage-" + strings.ReplaceAll(t.Name(), "/", "-")
	usage.RegisterNamedPlugin(pluginName, transcriptionUsagePluginFunc(func(_ context.Context, record usage.Record) {
		select {
		case records <- record:
		default:
		}
	}))
	t.Cleanup(func() {
		usage.RegisterNamedPlugin(pluginName, transcriptionUsagePluginFunc(func(context.Context, usage.Record) {}))
	})

	recorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{"model": "whisper-1"}, true))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}

	select {
	case record := <-records:
		if record.Provider != "codex" || record.Model != "whisper-1" || record.AuthID != "codex-oauth" || record.Failed {
			t.Fatalf("usage record = %#v, want successful Codex record", record)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for Codex transcription usage record")
	}
}

func TestCodexTranscriptionValidatesFileModelAndFormats(t *testing.T) {
	var upstreamCalls atomic.Int32
	handler, _, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"ok"}`))
	}, false)
	defer closeServer()

	tests := []struct {
		name          string
		fields        map[string]string
		includeFile   bool
		wantStatus    int
		wantSubstring string
	}{
		{name: "missing file", fields: map[string]string{"model": "whisper-1"}, wantStatus: http.StatusBadRequest, wantSubstring: "file is required"},
		{name: "unsupported model", fields: map[string]string{"model": "gpt-4.1"}, includeFile: true, wantStatus: http.StatusBadRequest, wantSubstring: "unsupported transcription model"},
		{name: "srt has no timestamps", fields: map[string]string{"model": "whisper-1", "response_format": "srt"}, includeFile: true, wantStatus: http.StatusBadRequest, wantSubstring: "does not provide timestamps"},
		{name: "vtt has no timestamps", fields: map[string]string{"model": "whisper-1", "response_format": "vtt"}, includeFile: true, wantStatus: http.StatusBadRequest, wantSubstring: "does not provide timestamps"},
		{name: "invalid temperature", fields: map[string]string{"model": "whisper-1", "temperature": "2"}, includeFile: true, wantStatus: http.StatusBadRequest, wantSubstring: "temperature must be"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, test.fields, test.includeFile))
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, test.wantStatus, recorder.Body.String())
			}
			if !strings.Contains(recorder.Body.String(), test.wantSubstring) {
				t.Fatalf("body = %q, want substring %q", recorder.Body.String(), test.wantSubstring)
			}
		})
	}
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0 for rejected requests", got)
	}
}

func TestCodexTranscriptionNormalizesTextAndVerboseJSON(t *testing.T) {
	handler, _, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, _ *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"normalized text"}`))
	}, false)
	defer closeServer()

	textRecorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":           "whisper-1",
		"response_format": "text",
	}, true))
	if textRecorder.Code != http.StatusOK || textRecorder.Body.String() != "normalized text" {
		t.Fatalf("text response = %d %q, want 200 normalized text", textRecorder.Code, textRecorder.Body.String())
	}
	if got := textRecorder.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Fatalf("text Content-Type = %q, want text/plain", got)
	}

	verboseRecorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":           "whisper-1",
		"response_format": "verbose_json",
	}, true))
	if verboseRecorder.Code != http.StatusOK {
		t.Fatalf("verbose_json status = %d; body=%s", verboseRecorder.Code, verboseRecorder.Body.String())
	}
	var verboseResponse map[string]any
	if errUnmarshal := json.Unmarshal(verboseRecorder.Body.Bytes(), &verboseResponse); errUnmarshal != nil {
		t.Fatalf("decode verbose_json response: %v", errUnmarshal)
	}
	if verboseResponse["text"] != "normalized text" {
		t.Fatalf("verbose_json text = %#v, want normalized text", verboseResponse["text"])
	}
}

func TestCodexTranscriptionConvertsUpstreamErrors(t *testing.T) {
	handler, _, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, _ *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		responseWriter.WriteHeader(http.StatusUnauthorized)
		_, _ = responseWriter.Write([]byte(`{"error":{"message":"invalid access token"}}`))
	}, false)
	defer closeServer()

	recorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{"model": "whisper-1"}, true))
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401; body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "invalid access token") {
		t.Fatalf("body = %q, want upstream error detail", recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "codex-token") {
		t.Fatalf("response leaked OAuth token: %q", recorder.Body.String())
	}
	gotAuth, ok := handler.authManager.GetByID("codex-oauth")
	if !ok || gotAuth == nil {
		t.Fatal("Codex auth was not retained by the manager")
	}
	if gotAuth.Success != 0 || gotAuth.Failed != 1 {
		t.Fatalf("auth totals = success=%d failed=%d, want 0/1", gotAuth.Success, gotAuth.Failed)
	}
}

func TestCodexTranscriptionRefreshesOAuthAfterUnauthorized(t *testing.T) {
	var calls atomic.Int32
	handler, executor, closeServer := newTranscriptionTestHandler(t, func(responseWriter http.ResponseWriter, request *http.Request) {
		call := calls.Add(1)
		if call == 1 {
			responseWriter.WriteHeader(http.StatusUnauthorized)
			return
		}
		if got := request.Header.Get("Authorization"); got != "Bearer refreshed-token" {
			t.Errorf("retry Authorization = %q, want refreshed token", got)
		}
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"after refresh"}`))
	}, true)
	defer closeServer()

	recorder := invokeTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{"model": "whisper-1"}, true))
	if recorder.Code != http.StatusOK || recorder.Body.String() != `{"text":"after refresh"}` {
		t.Fatalf("response = %d %q, want refreshed JSON response", recorder.Code, recorder.Body.String())
	}
	if calls.Load() != 2 || executor.refreshCalls.Load() != 1 {
		t.Fatalf("upstream calls = %d, refresh calls = %d, want 2 and 1", calls.Load(), executor.refreshCalls.Load())
	}
}

func TestAntigravityTranscriptionBuildsAudioRequestAndNormalizesJSON(t *testing.T) {
	handler, executor := newAntigravityTranscriptionTestHandler(t, []byte(`{"candidates":[{"content":{"parts":[{"thought":true,"text":"hidden reasoning"},{"text":"hello from Google"}]}}]}`), nil)
	recorder := invokeAntigravityTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":           "gemini-transcribe",
		"language":        "en",
		"prompt":          "Use the speaker's spelling.",
		"temperature":     "0",
		"response_format": "json",
	}, true))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if recorder.Body.String() != `{"text":"hello from Google"}` {
		t.Fatalf("body = %q, want normalized JSON", recorder.Body.String())
	}
	if executor.calls.Load() != 1 {
		t.Fatalf("executor calls = %d, want 1", executor.calls.Load())
	}
	if executor.selectedAuth == nil || executor.selectedAuth.Provider != "antigravity" {
		t.Fatalf("selected auth = %#v, want Antigravity auth", executor.selectedAuth)
	}
	if executor.request.Model != antigravityTranscriptionDefaultModel {
		t.Fatalf("upstream model = %q, want %q", executor.request.Model, antigravityTranscriptionDefaultModel)
	}
	if executor.options.SourceFormat != "gemini" || executor.options.ResponseFormat != "gemini" {
		t.Fatalf("formats = %q -> %q, want gemini -> gemini", executor.options.SourceFormat, executor.options.ResponseFormat)
	}

	var payload struct {
		Contents []struct {
			Parts []struct {
				Text       string `json:"text"`
				InlineData struct {
					MIMEType string `json:"mime_type"`
					Data     string `json:"data"`
				} `json:"inline_data"`
			} `json:"parts"`
		} `json:"contents"`
		GenerationConfig struct {
			Temperature float64 `json:"temperature"`
		} `json:"generationConfig"`
	}
	if errUnmarshal := json.Unmarshal(executor.request.Payload, &payload); errUnmarshal != nil {
		t.Fatalf("decode Antigravity payload: %v", errUnmarshal)
	}
	if len(payload.Contents) != 1 || len(payload.Contents[0].Parts) != 2 {
		t.Fatalf("payload contents = %#v, want one user turn with text and audio", payload.Contents)
	}
	if !strings.Contains(payload.Contents[0].Parts[0].Text, "language is en") || !strings.Contains(payload.Contents[0].Parts[0].Text, "speaker's spelling") {
		t.Fatalf("instruction = %q, want language and prompt", payload.Contents[0].Parts[0].Text)
	}
	inlineData := payload.Contents[0].Parts[1].InlineData
	if inlineData.MIMEType != "audio/wav" {
		t.Fatalf("audio MIME type = %q, want audio/wav", inlineData.MIMEType)
	}
	decoded, errDecode := base64.StdEncoding.DecodeString(inlineData.Data)
	if errDecode != nil || string(decoded) != "RIFF-test-audio" {
		t.Fatalf("decoded audio = %q, decode error = %v", decoded, errDecode)
	}
	if payload.GenerationConfig.Temperature != 0 {
		t.Fatalf("temperature = %v, want 0", payload.GenerationConfig.Temperature)
	}
	if strings.Contains(string(executor.request.Payload), "antigravity-token") {
		t.Fatal("payload leaked OAuth token")
	}
	gotAuth, ok := handler.authManager.GetByID("antigravity-oauth")
	if !ok || gotAuth == nil {
		t.Fatal("Antigravity auth was not retained by the manager")
	}
	if gotAuth.Success != 1 || gotAuth.Failed != 0 {
		t.Fatalf("auth totals = success=%d failed=%d, want 1/0", gotAuth.Success, gotAuth.Failed)
	}
}

func TestAntigravityTranscriptionNormalizesTextAndRejectsTimestamps(t *testing.T) {
	handler, executor := newAntigravityTranscriptionTestHandler(t, []byte(`{"candidates":[{"content":{"parts":[{"text":"plain transcript"}]}}]}`), nil)
	textRecorder := invokeAntigravityTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":           "google-transcribe",
		"response_format": "text",
	}, true))
	if textRecorder.Code != http.StatusOK || textRecorder.Body.String() != "plain transcript" {
		t.Fatalf("text response = %d %q, want 200 plain transcript", textRecorder.Code, textRecorder.Body.String())
	}
	if got := textRecorder.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Fatalf("Content-Type = %q, want text/plain", got)
	}

	timestampRecorder := invokeAntigravityTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model":           "gemini-transcribe",
		"response_format": "vtt",
	}, true))
	if timestampRecorder.Code != http.StatusBadRequest || !strings.Contains(timestampRecorder.Body.String(), "does not provide timestamps") {
		t.Fatalf("timestamp response = %d %q, want unsupported timestamp error", timestampRecorder.Code, timestampRecorder.Body.String())
	}
	if executor.calls.Load() != 1 {
		t.Fatalf("executor calls = %d, want timestamp request rejected before execution", executor.calls.Load())
	}
}

func TestAntigravityTranscriptionConvertsUpstreamErrors(t *testing.T) {
	handler, _ := newAntigravityTranscriptionTestHandler(t, nil, transcriptionStatusError{status: http.StatusUnauthorized, message: "invalid Antigravity access token"})
	recorder := invokeAntigravityTranscriptionHandler(t, handler, transcriptionRequest(t, map[string]string{
		"model": "gemini-3-flash-agent",
	}, true))
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "antigravity_auth_error") || !strings.Contains(recorder.Body.String(), "invalid Antigravity access token") {
		t.Fatalf("body = %q, want auth error details", recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "antigravity-token") {
		t.Fatal("response leaked OAuth token")
	}
}

func TestCodexTranscriptionRouteDoesNotReplaceRealtimeRoute(t *testing.T) {
	server := newTestServer(t)

	transcriptionRequest := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", nil)
	transcriptionRequest.Header.Set("Authorization", "Bearer test-key")
	transcriptionResponse := httptest.NewRecorder()
	server.engine.ServeHTTP(transcriptionResponse, transcriptionRequest)
	if transcriptionResponse.Code != http.StatusBadRequest {
		t.Fatalf("transcription route status = %d, want 400 after auth; body=%s", transcriptionResponse.Code, transcriptionResponse.Body.String())
	}

	realtimeRequest := httptest.NewRequest(http.MethodPost, "/v1/realtime/transcription_sessions", nil)
	realtimeRequest.Header.Set("Authorization", "Bearer test-key")
	realtimeResponse := httptest.NewRecorder()
	server.engine.ServeHTTP(realtimeResponse, realtimeRequest)
	if realtimeResponse.Code != http.StatusNotImplemented {
		t.Fatalf("realtime transcription route status = %d, want 501; body=%s", realtimeResponse.Code, realtimeResponse.Body.String())
	}
}
