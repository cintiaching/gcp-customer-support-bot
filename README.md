# gcp-customer-support-bot
A toy project to pick up GCP.

This repo contains a simplified version of the 
[LangGraph tutorial](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/customer-support/customer-support.ipynb) 
customer support assistant for Swiss Airlines.

Technology stack used:
- Vector Store: Firestore for data storage and vector search extension
- Text Embedding: Vertex AI Embedding for Text (text-embedding-005)
- Model: Vertex AI Gemini 2.0 Flash (gemini-1.5-pro-002)
- Deployment: Chat interface containerized and deployed on Google Kubernetes Engine

## Usage

Setup:
1. Install `google-cloud-sdk`
2. Install dependency
    ```shell
   uv sync
   ```
3. Login to project
    ```shell
   gcloud auth application-default login
   ```
