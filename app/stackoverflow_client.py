import requests
# ... existing imports and class definition ...

class StackOverflowClient:
    # ... existing methods (e.g., search for questions) ...

    def fetch_answers(self, question_id: int):
        """
        Fetch all answers for a given StackOverflow question ID.
        Returns a list of answer records with answer_id, question_id, body, etc.
        """
        url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
        params = {
            "order": "desc",
            "sort": "votes",       # get highest-voted answers first (or use 'activity' for latest)
            "site": "stackoverflow",
            "filter": "withbody"   # include the answer body in the response
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"StackOverflow API error: {response.status_code} - {response.text}")
        data = response.json()
        answers = []
        for item in data.get("items", []):
            answer_id = item.get("answer_id")
            body_html = item.get("body", "")  # answer body is in HTML
            # Optionally strip HTML tags from body_html:
            # plain_text = re.sub('<[^<]+?>', '', html.unescape(body_html))
            answers.append({
                "answer_id": answer_id,
                "question_id": question_id,
                "body": body_html,
                "is_accepted": item.get("is_accepted", False),
                "score": item.get("score", 0)
            })
        return answers