from recall.core.model_display import model_display_name


def test_model_display_name_gpt_oss():
    assert model_display_name("openai/gpt-oss-120b") == "GPT OSS 120B"
    assert model_display_name("openrouter/openai/gpt-oss-120b") == "GPT OSS 120B"


def test_model_display_name_generic():
    assert model_display_name("deepseek/deepseek-v4-flash-0731") == "Deepseek V4 Flash 0731"
