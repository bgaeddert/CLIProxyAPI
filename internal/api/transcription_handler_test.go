package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
)

type transcriptionTestExecutor struct {
	refreshCalls atomic.Int32
	httpCalls    atomic.Int32
}

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

func invokeTranscriptionHandler(t *testing.T, handler *codexTranscriptionHandler, request *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(recorder)
	context.Request = request
	handler.Handle(context)
	return recorder
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
