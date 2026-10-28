package responses

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func echoChunk(content string, extra string) []byte {
	return []byte(`data: {"id":"chatcmpl-1","created":1700000000,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{` + extra + `}}]}`)
}

func runEchoStream(request []byte) []byte {
	chunks := [][]byte{
		echoChunk("", `"role":"assistant","content":""`),
		echoChunk("", `"reasoning_content":"think"`),
		echoChunk("", `"content":"Hel"`),
		echoChunk("", `"content":"lo"`),
		[]byte(`data: {"id":"chatcmpl-1","created":1700000000,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"editor__read","arguments":"{\"path\":\"/tmp/x\"}"}}]}}]}`),
		[]byte(`data: {"id":"chatcmpl-1","created":1700000000,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":7}}`),
		[]byte(`data: [DONE]`),
	}
	var param any
	var out []byte
	for _, chunk := range chunks {
		out = append(out, bytes.Join(ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "test", request, request, chunk, &param), nil)...)
	}
	return out
}

func echoRequests() []struct {
	name string
	raw  []byte
} {
	history := strings.Repeat("context ", 4096)
	tools := `[{"type":"namespace","name":"editor","tools":[{"type":"function","name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}]`
	return []struct {
		name string
		raw  []byte
	}{
		{"nil", nil},
		{"empty-object", []byte(`{}`)},
		{"malformed", []byte(`{`)},
		{"full", []byte(`{"model":"gpt-5","instructions":"` + history + `","max_output_tokens":4096,"max_tool_calls":3,"parallel_tool_calls":false,"previous_response_id":"resp_prev","prompt_cache_key":"k1","reasoning":{"effort":"high","summary":"auto"},"safety_identifier":"user-1","service_tier":"priority","store":true,"temperature":1,"text":{"format":{"type":"json_schema","name":"x"}},"tool_choice":"auto","tools":` + tools + `,"top_logprobs":5,"top_p":0.5,"truncation":"auto","user":"u1","metadata":{"b":2,"a":1},"input":[{"role":"user","content":"hi"}]}`)},
		{"weird-types", []byte(`{"model":42,"instructions":null,"temperature":"hot","store":"yes","parallel_tool_calls":1,"max_output_tokens":"999","top_p":2,"tools":"none","metadata":[1,2],"user":{"id":7},"input":[]}`)},
		{"duplicate-keys", []byte(`{"model":"first","input":[],"model":"second","temperature":0.1,"temperature":0.9}`)},
		{"escaped-key", []byte("{\"mod\\u0065l\":\"esc-model\",\"input\":[],\"tools\":" + tools + "}")},
		{"max-tokens-only", []byte(`{"model":"gpt-5","max_tokens":777,"input":[]}`)},
	}
}

// Golden hashes cover request echo extraction and the preserved provider usage metadata.
var echoGoldenHashes = map[string][2]string{
	"nil":             {"32385f68115ca05269d70aecb6b84cb913d16eae5151eb1624a4c7884981477a", "25aeb82ad9908c1917a40c390ad10bf56796b2573d762227d7f07c0a4d0e3846"},
	"empty-object":    {"32385f68115ca05269d70aecb6b84cb913d16eae5151eb1624a4c7884981477a", "25aeb82ad9908c1917a40c390ad10bf56796b2573d762227d7f07c0a4d0e3846"},
	"malformed":       {"32385f68115ca05269d70aecb6b84cb913d16eae5151eb1624a4c7884981477a", "25aeb82ad9908c1917a40c390ad10bf56796b2573d762227d7f07c0a4d0e3846"},
	"full":            {"da9d0979eb12b0ebd467f3c832db970c25eb6fcff17a33d55dc2bfcf2de0059e", "a24d885bfc0c80fd8dcbcd25b2d012edd018727ede100492a086c917b51367c0"},
	"weird-types":     {"d92c0fae28c40b6b0a249eff1f735d846d3fddd79b3d82085ef2151c0f4e9a6a", "a8754e66915cd73ed5eedb6d0717eb3d31151c2fc7f482ae93fdf8ebfe618476"},
	"duplicate-keys":  {"690b195baa35f2f3733886ec2ec9767ce475ac0335c0a5c066f1fd0e881d47df", "d1691fa385ea3405b977956c1a8477a28a38f5908e00ff765878349580afaa1c"},
	"escaped-key":     {"81326debd48e496dbbb6f1f67697ab8e18477a985da02dc3d371da293f1d5b3a", "c856f1cfee5c287bcf946a2f42f7c01b32bc4f0b6ff4033403ab9df3326eeed7"},
	"max-tokens-only": {"755a6ddbc50d10e1c1fafe8047423f88395b38560c437fcc780fee3ba27bcd7a", "dd933d6e693b93970b728fe556f434e59965003095fd169f3e1908755cf6b0d2"},
}

func hashHex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestResponsesEchoGoldenOutputs(t *testing.T) {
	for _, tc := range echoRequests() {
		want := echoGoldenHashes[tc.name]
		first := runEchoStream(tc.raw)
		if got := hashHex(first); got != want[0] {
			t.Errorf("%s stream: hash %s, want %s", tc.name, got, want[0])
		}
		resp := []byte(`{"id":"chatcmpl-9","created":1700000000,"model":"resp-model","choices":[{"message":{"role":"assistant","content":"done","reasoning_content":"why","tool_calls":[{"id":"call_9","type":"function","function":{"name":"editor__read","arguments":"{\"path\":\"/tmp/y\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":4}}`)
		var p1 any
		n1 := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "test", tc.raw, tc.raw, resp, &p1)
		if got := hashHex(n1); got != want[1] {
			t.Errorf("%s nonstream: hash %s, want %s", tc.name, got, want[1])
		}
	}
}

func BenchmarkResponsesEchoShortStreams(b *testing.B) {
	request := responsesPerfRequest(100)
	chunk := echoChunk("", `"content":"x"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var param any
		ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "test", request, request, chunk, &param)
		ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "test", request, request, []byte("data: [DONE]"), &param)
	}
}

// Golden hashes for root-shape and reasoning-inclusion edge cases.
func TestResponsesEchoGoldenEdgeCases(t *testing.T) {
	for name, req := range map[string][]byte{
		"array-root":  []byte(`[{"model":"x"}]`),
		"scalar-root": []byte(`"hello"`),
		"number-root": []byte(`42`),
	} {
		if got := hashHex(runEchoStream(req)); got != "32385f68115ca05269d70aecb6b84cb913d16eae5151eb1624a4c7884981477a" {
			t.Errorf("%s stream: hash %s", name, got)
		}
		resp := []byte(`{"id":"c1","created":1700000000,"model":"m","choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
		var p any
		if got := hashHex(ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "test", req, req, resp, &p)); got != "dd692977eabc936838864cb4bad57311e9414956ba8362375ce324804afbea70" {
			t.Errorf("%s nonstream: hash %s", name, got)
		}
	}
	respNoRC := []byte(`{"id":"c2","created":1700000000,"model":"m","choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	want := map[string]string{
		"req-with-reasoning":    "62ddf67a2e7e2804407045b44f5d3d05f534b2699e3c1d45f497a2d277a23707",
		"req-without-reasoning": "59ce4f145470320095f8c5261c64c1f02727afc1ddcbfcfa9f4324ea735e64c5",
		"req-reasoning-null":    "16097b4434a3774873ea9a026656fd35bd4872ff119134d4073047caca245098",
	}
	for name, req := range map[string][]byte{
		"req-with-reasoning":    []byte(`{"model":"gpt-5","reasoning":{"effort":"low"},"input":[]}`),
		"req-without-reasoning": []byte(`{"model":"gpt-5","input":[]}`),
		"req-reasoning-null":    []byte(`{"model":"gpt-5","reasoning":null,"input":[]}`),
	} {
		var p any
		if got := hashHex(ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "test", req, req, respNoRC, &p)); got != want[name] {
			t.Errorf("%s nonstream: hash %s, want %s", name, got, want[name])
		}
	}
}
