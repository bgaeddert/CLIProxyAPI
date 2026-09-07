package executor

import (
	"bytes"
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func TestOpenAICompatExecutorForwardsTranscriptionMultipart(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("request = %s %s, want POST /v1/audio/transcriptions", request.Method, request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer provider-key" {
			t.Errorf("Authorization = %q, want provider key", got)
		}
		if errParse := request.ParseMultipartForm(1 << 20); errParse != nil {
			t.Errorf("parse multipart form: %v", errParse)
			return
		}
		if got := request.FormValue("model"); got != "upstream-model" {
			t.Errorf("model = %q, want upstream-model", got)
		}
		if got := request.FormValue("language"); got != "en" {
			t.Errorf("language = %q, want en", got)
		}
		if got := request.FormValue("response_format"); got != "json" {
			t.Errorf("response_format = %q, want json", got)
		}
		fileHeaders := request.MultipartForm.File["file"]
		if len(fileHeaders) != 1 {
			t.Errorf("file count = %d, want 1", len(fileHeaders))
		}
		responseWriter.Header().Set("Content-Type", "application/json")
		_, _ = responseWriter.Write([]byte(`{"text":"forwarded transcript"}`))
	}))
	defer server.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if errField := writer.WriteField("model", "client-model"); errField != nil {
		t.Fatalf("write model: %v", errField)
	}
	if errField := writer.WriteField("language", "en"); errField != nil {
		t.Fatalf("write language: %v", errField)
	}
	if errField := writer.WriteField("response_format", "json"); errField != nil {
		t.Fatalf("write response format: %v", errField)
	}
	part, errFile := writer.CreateFormFile("file", "sample.wav")
	if errFile != nil {
		t.Fatalf("create file: %v", errFile)
	}
	if _, errWrite := part.Write([]byte("RIFF-test-audio")); errWrite != nil {
		t.Fatalf("write file: %v", errWrite)
	}
	if errClose := writer.Close(); errClose != nil {
		t.Fatalf("close multipart writer: %v", errClose)
	}

	executor := NewOpenAICompatExecutor("openai-compatible-test", &config.Config{})
	response, errExecute := executor.Execute(context.Background(), &cliproxyauth.Auth{
		ID:       "openai-compat-auth",
		Provider: "openai-compatible-test",
		Attributes: map[string]string{
			"base_url": server.URL + "/v1",
			"api_key":  "provider-key",
		},
	}, cliproxyexecutor.Request{
		Model:   "upstream-model",
		Payload: body.Bytes(),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-transcription"),
		Headers: http.Header{
			"Content-Type": []string{writer.FormDataContentType()},
		},
	})
	if errExecute != nil {
		t.Fatalf("execute transcription: %v", errExecute)
	}
	if string(response.Payload) != `{"text":"forwarded transcript"}` {
		t.Fatalf("response payload = %q, want upstream JSON", response.Payload)
	}
}
