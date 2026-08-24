package handlers

import (
	"context"
	"net/http"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

func TestBeginRequestLifecyclePublishesCompletionOnce(t *testing.T) {
	var completions []pluginapi.RequestCompletion
	host := &handlerInterceptorTestHost{
		completeRequest: func(_ context.Context, completion pluginapi.RequestCompletion) {
			completions = append(completions, completion)
		},
	}
	handler := NewBaseAPIHandlers(&config.SDKConfig{}, nil)
	handler.SetPluginHost(host)

	lifecycle := handler.BeginRequestLifecycle(context.Background(), "openai", "whisper-1", "whisper-1", map[string]any{
		"request_path": "/v1/audio/transcriptions",
	})
	lifecycle.Complete(http.StatusOK)
	lifecycle.Complete(http.StatusBadGateway)

	if len(completions) != 1 {
		t.Fatalf("completion count = %d, want 1", len(completions))
	}
	completion := completions[0]
	if completion.Outcome != pluginapi.RequestCompletionSucceeded || completion.StatusCode != http.StatusOK {
		t.Fatalf("completion = %#v, want succeeded/200", completion)
	}
	if completion.RequestID == "" || completion.StartedAt.IsZero() || completion.CompletedAt.IsZero() {
		t.Fatalf("completion missing lifecycle metadata: %#v", completion)
	}
}
