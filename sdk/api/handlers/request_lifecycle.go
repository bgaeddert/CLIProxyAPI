package handlers

import (
	"net/http"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
	"golang.org/x/net/context"
)

// RequestLifecycle is the shared request-completion hook used by handlers that
// cannot use the standard JSON execution helpers but still need to report the
// request to plugin and dashboard observers.
type RequestLifecycle struct {
	tracker *requestLifecycleTracker
}

// BeginRequestLifecycle starts a request lifecycle record.
func (h *BaseAPIHandler) BeginRequestLifecycle(ctx context.Context, sourceFormat, model, requestedModel string, metadata map[string]any) *RequestLifecycle {
	if h == nil {
		return nil
	}
	return &RequestLifecycle{
		tracker: h.newRequestLifecycleTracker(ctx, sourceFormat, model, requestedModel, false, metadata, ""),
	}
}

// Complete records the final status of a request.
func (l *RequestLifecycle) Complete(statusCode int) {
	if l == nil || l.tracker == nil {
		return
	}
	if statusCode <= 0 {
		statusCode = http.StatusInternalServerError
	}

	outcome := pluginapi.RequestCompletionSucceeded
	if statusCode >= http.StatusInternalServerError {
		outcome = pluginapi.RequestCompletionFailed
	} else if statusCode >= http.StatusBadRequest {
		outcome = pluginapi.RequestCompletionRejected
	}
	l.tracker.complete(outcome, statusCode, nil)
}
