package cliproxy

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
)

func TestBuildOpenAICompatibilityConfigModels_InputModalities(t *testing.T) {
	compat := &config.OpenAICompatibility{
		Name: "mimo",
		Models: []config.OpenAICompatibilityModel{
			{
				Name:            "upstream-vision",
				Alias:           "mimo-v2.5-pro",
				DisplayName:     "Mimo Vision",
				InputModalities: []string{"TEXT", "image", "image"},
			},
			{
				Name:  "upstream-image",
				Alias: "compat-image",
				Image: true,
			},
			{
				Name:          "upstream-transcription",
				Alias:         "compat-transcription",
				Transcription: true,
			},
		},
	}

	models := buildOpenAICompatibilityConfigModels(compat)
	if len(models) != 3 {
		t.Fatalf("model count = %d, want 3", len(models))
	}

	var vision *ModelInfo
	var imageModel *ModelInfo
	var transcriptionModel *ModelInfo
	for _, model := range models {
		if model == nil {
			continue
		}
		switch model.ID {
		case "mimo-v2.5-pro":
			vision = model
		case "compat-image":
			imageModel = model
		case "compat-transcription":
			transcriptionModel = model
		}
	}
	if vision == nil {
		t.Fatal("expected vision model")
	}
	if vision.DisplayName != "Mimo Vision" {
		t.Fatalf("DisplayName = %q, want Mimo Vision", vision.DisplayName)
	}
	if got := joinModalities(vision.SupportedInputModalities); got != "text,image" {
		t.Fatalf("SupportedInputModalities = %q, want text,image", got)
	}
	if imageModel == nil {
		t.Fatal("expected image model")
	}
	if imageModel.DisplayName != "compat-image" {
		t.Fatalf("image DisplayName = %q, want compat-image", imageModel.DisplayName)
	}
	if imageModel.Type != registry.OpenAIImageModelType {
		t.Fatalf("image model type = %q, want %q", imageModel.Type, registry.OpenAIImageModelType)
	}
	if len(imageModel.SupportedInputModalities) != 0 {
		t.Fatalf("image model input modalities = %+v, want none", imageModel.SupportedInputModalities)
	}
	if !imageModel.SupportsImageEndpoints || imageModel.SupportsTranscriptionEndpoints {
		t.Fatalf("image endpoint capabilities = image:%t transcription:%t, want true:false", imageModel.SupportsImageEndpoints, imageModel.SupportsTranscriptionEndpoints)
	}
	if transcriptionModel == nil {
		t.Fatal("expected transcription model")
	}
	if transcriptionModel.Type != registry.OpenAITranscriptionModelType {
		t.Fatalf("transcription model type = %q, want %q", transcriptionModel.Type, registry.OpenAITranscriptionModelType)
	}
	if !transcriptionModel.SupportsTranscriptionEndpoints || transcriptionModel.SupportsImageEndpoints {
		t.Fatalf("transcription endpoint capabilities = image:%t transcription:%t, want false:true", transcriptionModel.SupportsImageEndpoints, transcriptionModel.SupportsTranscriptionEndpoints)
	}
	if got := joinModalities(transcriptionModel.SupportedInputModalities); got != "audio" {
		t.Fatalf("transcription input modalities = %q, want audio", got)
	}
	if got := joinModalities(transcriptionModel.SupportedOutputModalities); got != "text" {
		t.Fatalf("transcription output modalities = %q, want text", got)
	}
}

func joinModalities(modalities []string) string {
	if len(modalities) == 0 {
		return ""
	}
	out := modalities[0]
	for i := 1; i < len(modalities); i++ {
		out += "," + modalities[i]
	}
	return out
}
