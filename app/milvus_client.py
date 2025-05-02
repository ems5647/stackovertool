from pymilvus import connections,Collection, CollectionSchema, FieldSchema, DataType

class MilvusClient:
    def __init__(self, collection_name: str, embedding_dim: int):
        self.collection_name = collection_name
        connections.connect(alias="default", host="localhost", port="19530")
        # Define the schema with additional fields for Q&A
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Milvus primary key
            FieldSchema(name="source_id", dtype=DataType.INT64, description="StackOverflow question/answer ID"),
            FieldSchema(name="parent_id", dtype=DataType.INT64, description="Parent question ID (0 if entry is a question)"),
            FieldSchema(name="is_question", dtype=DataType.BOOL, description="True if question, False if answer"),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255, description="Question title (empty for answers)"),
            FieldSchema(name="body", dtype=DataType.VARCHAR, max_length=10000, description="Post body text (question or answer)"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        schema = CollectionSchema(fields, description="StackOverflow Q&A embeddings")
        # Create or load the collection
        self.collection = Collection(name=self.collection_name, schema=schema, using="default", consistency_level="Strong")
        # You may need to create an index on the embedding field if not already exist
        # e.g., self.collection.create_index(field_name="embedding", index_params={...})
        # Also consider indexing source_id for faster lookups if needed (Milvus might allow scalar indexing).
    
    def insert_entry(self, source_id: int, parent_id: int, is_question: bool, title: str, body: str, embedding: list[float]):
        """
        Insert a single question or answer entry into Milvus.
        """
        # Prepare data for each field (excluding the auto_id primary key)
        data = [
            [source_id],
            [parent_id],
            [is_question],
            [title],
            [body],
            [embedding]
        ]
        # Insert the data into the collection
        insert_result = self.collection.insert(data)
        return insert_result

    def entry_exists(self, source_id: int, is_question: bool) -> bool:
        """
        Check if an entry (question or answer) with the given StackOverflow ID already exists in the collection.
        """
        # Milvus boolean values in queries: use true/false lowercase
        expr = f"source_id == {source_id} and is_question == {str(is_question).lower()}"
        result = self.collection.query(expr, output_fields=["source_id"], limit=1)
        return len(result) > 0

    def has_answers_for_question(self, question_id: int) -> bool:
        """
        Check if any answers for the given question ID are stored in the collection.
        """
        expr = f"parent_id == {question_id}"
        result = self.collection.query(expr, output_fields=["source_id"], limit=1)
        return len(result) > 0

    def search_vectors(self, embedding_vector: list[float], top_k: int = 5):
        """
        Search the collection for the most similar entries to the given embedding vector.
        Returns a list of hits with their fields.
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # example uses Inner Product; adjust as needed
        results = self.collection.search(
            [embedding_vector],
            "embedding",
            param=search_params,
            limit=top_k,
            output_fields=["source_id", "parent_id", "is_question", "title", "body"]
        )
        hits = results[0]  # results for the single query vector
        hits_data = []
        for hit in hits:
            hits_data.append({
                "score": hit.distance,  # similarity score (distance)
                "source_id": int(hit.entity.get("source_id")),
                "parent_id": int(hit.entity.get("parent_id")),
                "is_question": bool(hit.entity.get("is_question")),
                "title": hit.entity.get("title"),
                "body": hit.entity.get("body")
            })
        return hits_data