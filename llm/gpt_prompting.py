from openai import OpenAI
import json
from retrying import retry


def prompting(model_version, api_key, prompt, text):
    client = OpenAI(
        api_key = api_key
    )
    response = client.chat.completions.create(
    model = model_version,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt + text}
    ],
    temperature=0,
    )
    result = response.choices[0].message.content
    result = result.replace('"', '').replace(' ', '')
    
    return result


def print_retry_details(exception):
    print(f"Error happened: {exception}")
    print("Retrying...")
    return True


class PromptingProcessor:
    def __init__(self, prompt, output_dir, output_name="prompting_results", model="chatgpt", model_version="gpt-3.5-turbo", api_key=""):
        self.prompt = prompt
        self.output_dir = output_dir
        self.output_name = output_name
        self.model = model
        self.model_version = model_version
        self.api_key = api_key

    @retry(retry_on_exception=print_retry_details)
    def process_one_text(self, text):
        if self.model == "chatgpt":
            result = prompting(self.model_version, self.api_key, self.prompt, text)
            return result

    def process_texts(self, texts, encoding="utf-8-sig"):
        output_path = self.output_dir + self.output_name + "_" + self.model + "_" + self.model_version + ".json"
        try:
            with open(output_path, "r", encoding=encoding) as f:
                results = json.load(f)
                print(f"{len(results)} results loaded from " + output_path)
        except:
            results = {}
        start_id = len(results)
        texts = texts[start_id:]
        for i, text in enumerate(texts, start=start_id):
            results[i] = self.process_one_text(text)
            print("Saving result " + str(i) + " to " + output_path)
            with open(output_path, "w", encoding=encoding) as f:
                json.dump(results, f, ensure_ascii=False)
        return results


