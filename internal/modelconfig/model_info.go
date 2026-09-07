package modelconfig

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
)

// ResolveModelInfo returns a private capability snapshot for a configured model.
// Static capabilities come from the suffix-free upstream name, while explicit
// configuration takes precedence.
func ResolveModelInfo(name, modelType string, support *registry.ThinkingSupport) *registry.ModelInfo {
	trimmedName := strings.TrimSpace(name)
	baseName := strings.TrimSpace(thinking.ParseSuffix(trimmedName).ModelName)
	info := registry.LookupStaticModelInfo(baseName)
	if info == nil {
		info = &registry.ModelInfo{}
	}
	info.ID = trimmedName
	info.Type = strings.TrimSpace(modelType)
	if support != nil {
		info.Thinking = NormalizeThinkingSupport(support)
	}
	info.UserDefined = false
	return info
}

// ApplyOpenAICompatibilityCapabilities applies the endpoint capabilities declared
// for a configured OpenAI-compatible model to its registry snapshot.
func ApplyOpenAICompatibilityCapabilities(info *registry.ModelInfo, model config.OpenAICompatibilityModel) {
	if info == nil {
		return
	}

	info.Type = registry.OpenAICompatibilityModelType
	info.SupportsImageEndpoints = model.Image
	info.SupportsTranscriptionEndpoints = model.Transcription
	switch {
	case model.Image && !model.Transcription:
		info.Type = registry.OpenAIImageModelType
	case model.Transcription && !model.Image:
		info.Type = registry.OpenAITranscriptionModelType
	}

	var requiredInput, requiredOutput []string
	if model.Transcription {
		requiredInput = []string{"audio"}
		requiredOutput = []string{"text"}
	}
	info.SupportedInputModalities = EnsureModalities(model.InputModalities, requiredInput...)
	info.SupportedOutputModalities = EnsureModalities(model.OutputModalities, requiredOutput...)
}

// EnsureModalities normalizes modality names and appends required non-empty values.
func EnsureModalities(raw []string, required ...string) []string {
	out := make([]string, 0, len(raw)+len(required))
	seen := make(map[string]struct{}, len(raw)+len(required))
	appendValue := func(value string) {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" {
			return
		}
		if _, exists := seen[value]; exists {
			return
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	for _, value := range raw {
		appendValue(value)
	}
	for _, value := range required {
		appendValue(value)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// NormalizeThinkingSupport clones and normalizes configured reasoning levels.
func NormalizeThinkingSupport(raw *registry.ThinkingSupport) *registry.ThinkingSupport {
	if raw == nil {
		return nil
	}
	normalized := *raw
	normalized.Levels = nil
	seen := make(map[string]struct{}, len(raw.Levels))
	for _, value := range raw.Levels {
		level := strings.ToLower(strings.TrimSpace(value))
		if level == "" {
			continue
		}
		switch level {
		case "none":
			normalized.ZeroAllowed = true
		case "auto":
			normalized.DynamicAllowed = true
		}
		if _, exists := seen[level]; exists {
			continue
		}
		seen[level] = struct{}{}
		normalized.Levels = append(normalized.Levels, level)
	}
	return &normalized
}
