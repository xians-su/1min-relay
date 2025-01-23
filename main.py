from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
from waitress import serve
import json
import tiktoken
import socket
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

# Function to ensure the storage directory exists
def check_if_storage_folder_exists():
    if not os.path.exists("storage"):
        os.makedirs("storage")


def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""
    
    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        tokens = tokenizer.encode(sentence)
        return len(tokens)

    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Use OpenAI's tiktoken for GPT models
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)

    else:
        # Default to openai
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)
app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app
)

ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features?isStreaming=true"

ONE_MIN_AVAILABLE_MODELS = ["mistral-nemo", "gpt-4o", "deepseek-chat"] # Models must be in the api name.
EXTERNAL_AVAILABLE_MODELS = [] # external API models
EXTERNAL_URL = "https://api.openai.com/v1/chat/completions" # Redirect URL to an external API if not available in 1 min. API MUST BE IN OPENAI STRUCTURE API.
EXTERNAL_API_KEY = "" # API key to access the external
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(ONE_MIN_AVAILABLE_MODELS)
AVAILABLE_MODELS.extend(EXTERNAL_AVAILABLE_MODELS)
PERMIT_MODELS_NOT_IN_AVAILABLE_MODELS = False

@app.route('/v1/models')
@limiter.limit("500 per minute")
def models():
    # Dynamically create the list of models with additional fields
    models_data = []
    one_min_models_data = [
        {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
        for model_name in ONE_MIN_AVAILABLE_MODELS
    ]
    hugging_models_data = [
        {"id": model_name, "object": "model", "owned_by": "Hugging Face"}
        for model_name in EXTERNAL_AVAILABLE_MODELS
    ]
    models_data.extend(one_min_models_data)
    models_data.extend(hugging_models_data)
    return jsonify({"data": models_data, "object": "list"})

def create_convo(api):
    headers = {
        "API-KEY": api,
        "Content-Type": "application/json"
    }
    data = {
        "title": "New Managed Conversation",
        "type": "CHAT_WITH_AI",
    }
    response = requests.post(ONE_MIN_CONVERSATION_API_URL, headers=headers, data=json.dumps(data))
    return response.json()

def ERROR_HANDLER(code, model=None, key=None):
    error_codes = {
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.", "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
    }
    # Return the error in a openai format
    error_data = {k: v for k, v in error_codes.get(code, {"message": "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"}
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400)

def format_conversation_history(messages, new_input):
    """
    Formats the conversation history into a structured string.
    
    Args:
        messages (list): List of message dictionaries from the request
    
    Returns:
        str: Formatted conversation history
    """
    formatted_history = ["Conversation History:\n"]
    for message in messages:
        role = message.get('role', '').capitalize()
        content = message.get('content', '')
        
        # Handle potential list content
        if isinstance(content, list):
            content = '\n'.join(item['text'] for item in content)
        
        formatted_history.append(f"{role}: {content}")
    formatted_history.append("Respond like normal. The conversation history will be automatically updated on the next MESSAGE. DO NOT ADD User: or Assistant: to your output. Just respond like normal.")
    formatted_history.append("User Message:\n" + new_input)
    
    return '\n'.join(formatted_history)

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None}}), 401

    api_key = auth_header.split(" ")[1]
    
    request_data = request.json
    
    all_messages = format_conversation_history(request_data.get('messages', []), request_data.get('new_input', ''))

    messages = request_data.get('messages', [])
    if not messages:
        return jsonify({"error": {"message": "No messages provided", "type": "invalid_request_error", "param": "messages", "code": None}}), 400

    user_input = messages[-1].get('content')
    if not user_input:
        return jsonify({"error": {"message": "No content in the last message", "type": "invalid_request_error", "param": "messages", "code": None}}), 400

    # Check if user_input is a list and combine text if necessary
    if isinstance(user_input, list):
        combined_text = '\n'.join(item['text'] for item in user_input)
        user_input = str(combined_text)

    prompt_token = calculate_token(str(user_input))
    if not PERMIT_MODELS_NOT_IN_AVAILABLE_MODELS and request_data.get('model', 'mistral-nemo') not in AVAILABLE_MODELS:
        return ERROR_HANDLER(1002, request_data.get('model', 'mistral-nemo'))

    payload = {
        "type": "CHAT_WITH_AI",
        "model": request_data.get('model', 'mistral-nemo'),
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": False
        }
    }
    
    headers = {"API-KEY": api_key, 'Content-Type': 'application/json'}

    if not request_data.get('stream', False):
        print('NON-STREAM')
        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        one_min_response = response.json()
        
        transformed_response = transform_response(one_min_response, request_data, prompt_token)
        response = make_response(jsonify(transformed_response))
        set_response_headers(response)
        
        return response, 200
    
    else:
        print('STREAM')
        response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, data=json.dumps(payload), headers=headers, stream=True)
        if response_stream.status_code != 200:
            if response_stream.status_code == 401:
                return ERROR_HANDLER(1020)
            return ERROR_HANDLER(response_stream.status_code)
        return Response(actual_stream_response(response_stream, request_data), content_type='text/event-stream')
def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204

def transform_response(one_min_response, request_data, prompt_token):
    completion_token = calculate_token(one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0])
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0],
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token
        }
    }
    
def set_response_headers(response ):
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access -Control-Allow-Origin'] = '*'
    response.headers['X-Request-ID'] = str (uuid.uuid4())

def stream_response(content, request_data):
    words = content.split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo'),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": word + " "
                    },
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        time.sleep(0.05)
    yield "data: [DONE]\n\n"

def actual_stream_response(response, request_data):
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None

        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo'),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.decode('utf-8')
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
    # Final chunk when iteration stops
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "delta": "",
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == '__main__':
    internal_ip = socket.gethostbyname(socket.gethostname())
    print('\n\nServer is ready to serve at:')
    print('Internal IP: ' + internal_ip + ':5001')
    print('\nEnter this url to OpenAI clients supporting custom endpoint:')
    print(internal_ip + ':5001/v1')
    print('If does not work, try:')
    print(internal_ip + ':5001/v1/chat/completions')
    serve(app, host='0.0.0.0', port=5001, threads=6)
