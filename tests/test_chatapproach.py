import json

import pytest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from openai.types.chat import ChatCompletion

from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach

from .mocks import (
    MOCK_EMBEDDING_DIMENSIONS,
    MOCK_EMBEDDING_MODEL_NAME,
    MockAsyncSearchResultsIterator,
)


async def mock_search(*args, **kwargs):
    return MockAsyncSearchResultsIterator(kwargs.get("search_text"), kwargs.get("vector_queries"))


@pytest.fixture
def chat_approach():
    return ChatReadRetrieveReadApproach(
        search_client=None,
        auth_helper=None,
        openai_client=None,
        chatgpt_model="gpt-35-turbo",
        chatgpt_deployment="chat",
        embedding_deployment="embeddings",
        embedding_model=MOCK_EMBEDDING_MODEL_NAME,
        embedding_dimensions=MOCK_EMBEDDING_DIMENSIONS,
        sourcepage_field="",
        content_field="",
        query_language="en-us",
        query_speller="lexicon",
    )


def test_get_search_query(chat_approach):
    payload = """
    {
	"id": "chatcmpl-81JkxYqYppUkPtOAia40gki2vJ9QM",
	"object": "chat.completion",
	"created": 1695324963,
	"model": "gpt-35-turbo",
	"prompt_filter_results": [
		{
			"prompt_index": 0,
			"content_filter_results": {
				"hate": {
					"filtered": false,
					"severity": "safe"
				},
				"self_harm": {
					"filtered": false,
					"severity": "safe"
				},
				"sexual": {
					"filtered": false,
					"severity": "safe"
				},
				"violence": {
					"filtered": false,
					"severity": "safe"
				}
			}
		}
	],
	"choices": [
		{
			"index": 0,
			"finish_reason": "function_call",
			"message": {
				"content": "this is the query",
				"role": "assistant",
				"tool_calls": [
					{
                        "id": "search_sources1235",
						"type": "function",
						"function": {
							"name": "search_sources",
							"arguments": "{\\n\\"search_query\\":\\"accesstelemedicineservices\\"\\n}"
						}
					}
				]
			},
			"content_filter_results": {

			}
		}
	],
	"usage": {
		"completion_tokens": 19,
		"prompt_tokens": 425,
		"total_tokens": 444
	}
}
"""
    default_query = "hello"
    chatcompletions = ChatCompletion.model_validate(json.loads(payload), strict=False)
    query = chat_approach.get_search_query(chatcompletions, default_query)

    assert query == "accesstelemedicineservices"


def test_get_search_query_returns_default(chat_approach):
    payload = '{"id":"chatcmpl-81JkxYqYppUkPtOAia40gki2vJ9QM","object":"chat.completion","created":1695324963,"model":"gpt-35-turbo","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}],"choices":[{"index":0,"finish_reason":"function_call","message":{"content":"","role":"assistant"},"content_filter_results":{}}],"usage":{"completion_tokens":19,"prompt_tokens":425,"total_tokens":444}}'
    default_query = "hello"
    chatcompletions = ChatCompletion.model_validate(json.loads(payload), strict=False)
    query = chat_approach.get_search_query(chatcompletions, default_query)

    assert query == default_query


def test_get_messages_from_history(chat_approach):
    messages = chat_approach.get_messages_from_history(
        system_prompt="You are a bot.",
        model_id="gpt-35-turbo",
        history=[
            {"role": "user", "content": "What does the first clause of the contract say?"},
            {
                "role": "assistant",
                "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
            },
            {"role": "user", "content": "What is the subject of the contract?"},
        ],
        user_content="What is the subject of the contract?",
        max_tokens=3000,
    )
    assert messages == [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "What does the first clause of the contract say?"},
        {
            "role": "assistant",
            "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
        },
        {"role": "user", "content": "What is the subject of the contract?"},
    ]


def test_get_messages_from_history_truncated(chat_approach):
    messages = chat_approach.get_messages_from_history(
        system_prompt="You are a bot.",
        model_id="gpt-35-turbo",
        history=[
            {"role": "user", "content": "What does the first clause of the contract say?"},
            {
                "role": "assistant",
                "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
            },
            {"role": "user", "content": "What is the subject of the contract?"},
        ],
        user_content="What is the subject of the contract?",
        max_tokens=10,
    )
    assert messages == [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "What is the subject of the contract?"},
    ]


def test_get_messages_from_history_truncated_longer(chat_approach):
    messages = chat_approach.get_messages_from_history(
        system_prompt="You are a bot.",  # 8 tokens
        model_id="gpt-35-turbo",
        history=[
            {"role": "user", "content": "What does the first clause of the contract say?"},  # 10 tokens
            {
                "role": "assistant",
                "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
            },  # 102 tokens
            {"role": "user", "content": "What is the payment method?"},  # 9 tokens
            {
                "role": "assistant",
                "content": "The payment will be made via deposit in bank account held by the supplier company. [contract02-2024.pdf]",
            },  # 26 tokens
            {"role": "user", "content": "What is the subject of the contract?"},  # 10 tokens
        ],
        user_content="What is the subject of the contract?",
        max_tokens=55,
    )
    assert messages == [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "What is the payment method?"},
        {
            "role": "assistant",
            "content": "The payment will be made via deposit in bank account held by the supplier company. [contract02-2024.pdf]",
        },
        {"role": "user", "content": "What is the subject of the contract?"},
    ]


def test_get_messages_from_history_truncated_break_pair(chat_approach):
    """Tests that the truncation breaks the pair of messages."""
    messages = chat_approach.get_messages_from_history(
        system_prompt="You are a bot.",  # 8 tokens
        model_id="gpt-35-turbo",
        history=[
            {"role": "user", "content": "What does the first clause of the contract say?"},  # 10 tokens
            {
                "role": "assistant",
                "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
            },  # 87 tokens
            {"role": "user", "content": "What is the payment method?"},  # 9 tokens
            {
                "role": "assistant",
                "content": "The payment will be made via deposit in bank account held by the supplier company. [contract02-2024.pdf]",
            },  # 26 tokens
            {"role": "user", "content": "What is the subject of the contract?"},  # 10 tokens
        ],
        user_content="What is the subject of the contract?",
        max_tokens=147,
    )
    assert messages == [
        {"role": "system", "content": "You are a bot."},
        {
            "role": "assistant",
            "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
        },
        {"role": "user", "content": "What is the payment method?"},
        {
            "role": "assistant",
            "content": "The payment will be made via deposit in bank account held by the supplier company. [contract02-2024.pdf]",
        },
        {"role": "user", "content": "What is the subject of the contract?"},
    ]


def test_get_messages_from_history_system_message(chat_approach):
    """Tests that the system message token count is considered."""
    messages = chat_approach.get_messages_from_history(
        system_prompt="Assistant helps the ATRA employees with their contracts questions. Be brief in your answers.",  # 24 tokens
        model_id="gpt-35-turbo",
        history=[
            {"role": "user", "content": "What does the first clause of the contract say?"},  # 10 tokens
            {
                "role": "assistant",
                "content": "The first clause of the contract deals with hiring of a specialized company to provide ongoing cleaning, sanitization, preservation, and hospital disinfection services at CISCOPAR unit premises. This encompasses not just labor but also the provision of materials, equipment, uniforms, and personal protective equipment (PPE), in accordance with the conditions, specifications, values, and consumption estimates outlined in the document. [contract02-2024.pdf].",
            },  # 102 tokens
            {"role": "user", "content": "What is the payment method?"},  # 9 tokens
            {
                "role": "assistant",
                "content": "The payment will be made via deposit in bank account held by the supplier company. [contract02-2024.pdf]",
            },  # 26 tokens
            {"role": "user", "content": "What is the subject of the contract?"},  # 10 tokens
        ],
        user_content="What is the subject of the contract?",
        max_tokens=36,
    )
    assert messages == [
        {
            "role": "system",
            "content": "Assistant helps the ATRA employees with their contracts questions. Be brief in your answers.",
        },
        {"role": "user", "content": "What is the subject of the contract?"},
    ]


def test_extract_followup_questions(chat_approach):
    content = "Here is answer to your question.<<What is the dress code?>>"
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question."
    assert followup_questions == ["What is the dress code?"]


def test_extract_followup_questions_three(chat_approach):
    content = """Here is answer to your question.

<<What are some examples of successful product launches they should have experience with?>>
<<Are there any specific technical skills or certifications required for the role?>>
<<Is there a preference for candidates with experience in a specific industry or sector?>>"""
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question.\n\n"
    assert followup_questions == [
        "What are some examples of successful product launches they should have experience with?",
        "Are there any specific technical skills or certifications required for the role?",
        "Is there a preference for candidates with experience in a specific industry or sector?",
    ]


def test_extract_followup_questions_no_followup(chat_approach):
    content = "Here is answer to your question."
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == "Here is answer to your question."
    assert followup_questions == []


def test_extract_followup_questions_no_pre_content(chat_approach):
    content = "<<What is the dress code?>>"
    pre_content, followup_questions = chat_approach.extract_followup_questions(content)
    assert pre_content == ""
    assert followup_questions == ["What is the dress code?"]


def test_get_messages_from_history_few_shots(chat_approach):
    user_query_request = "What is the subject of the contract?"
    messages = chat_approach.get_messages_from_history(
        system_prompt=chat_approach.query_prompt_template,
        model_id=chat_approach.chatgpt_model,
        user_content=user_query_request,
        history=[],
        max_tokens=chat_approach.chatgpt_token_limit - len(user_query_request),
        few_shots=chat_approach.query_prompt_few_shots,
    )
    # Make sure messages are in the right order
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[4]["role"] == "assistant"
    assert messages[5]["role"] == "user"
    assert messages[5]["content"] == user_query_request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "minimum_search_score,minimum_reranker_score,expected_result_count",
    [
        (0, 0, 1),
        (0, 2, 1),
        (0.03, 0, 1),
        (0.03, 2, 1),
        (1, 0, 0),
        (0, 4, 0),
        (1, 4, 0),
    ],
)
async def test_search_results_filtering_by_scores(
    monkeypatch, minimum_search_score, minimum_reranker_score, expected_result_count
):

    chat_approach = ChatReadRetrieveReadApproach(
        search_client=SearchClient(endpoint="", index_name="", credential=AzureKeyCredential("")),
        auth_helper=None,
        openai_client=None,
        chatgpt_model="gpt-35-turbo",
        chatgpt_deployment="chat",
        embedding_deployment="embeddings",
        embedding_model=MOCK_EMBEDDING_MODEL_NAME,
        embedding_dimensions=MOCK_EMBEDDING_DIMENSIONS,
        sourcepage_field="",
        content_field="",
        query_language="en-us",
        query_speller="lexicon",
    )

    monkeypatch.setattr(SearchClient, "search", mock_search)

    filtered_results = await chat_approach.search(
        top=10,
        query_text="test query",
        filter=None,
        vectors=[],
        use_semantic_ranker=True,
        use_semantic_captions=True,
        minimum_search_score=minimum_search_score,
        minimum_reranker_score=minimum_reranker_score,
    )

    assert (
        len(filtered_results) == expected_result_count
    ), f"Expected {expected_result_count} results with minimum_search_score={minimum_search_score} and minimum_reranker_score={minimum_reranker_score}"
