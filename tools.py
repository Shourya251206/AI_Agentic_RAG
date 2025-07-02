import json
import logging
import os
from typing import Optional, Set, Tuple
from azure.ai.ml import MLClient

# from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.identity import DefaultAzureCredential
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential


def process_citations(text_msg) -> str:
    """
    Given a text_message with .text.value and .text.annotations, append a
    **Citations** section with unique URL citations.
    """
    base = text_msg.text.value
    seen: Set[Tuple[str, str]] = set()

    for annot in getattr(text_msg.text, "annotations", []):
        uc = getattr(annot, "url_citation", None)
        if uc and uc.url:
            seen.add((annot.text, uc.url))

    if seen:
        base += "\n\n**Citations:**\n"
        for quote, url in seen:
            base += f"- **Quote**: {quote}  \n"
            base += f"  **URL**: [{url}]({url})\n"

    return base


def run_agent(
    project_client, agent_id: str, user_input: str
) -> Tuple[str, str]:
    """
    Mock implementation for agent testing. Returns canned responses for each agent type.
    """
    if agent_id == "FabricDataRetrievalAgent":
        # TODO: Optionally parse dataset name from user_input
        dataset_name = "your-dataset-name"  # <-- Replace with your actual dataset name
        try:
            dataset = project_client.data.get(name=dataset_name)
            return f"✅ Found dataset '{dataset_name}': {dataset.path}", ""
        except Exception as e:
            return f"❌ Error retrieving dataset '{dataset_name}': {e}", ""
    elif agent_id == "SharePointDataRetrievalAgent":
        site_url = "https://yourtenant.sharepoint.com/sites/yoursite"  # <-- Replace with your site URL
        username = "your-username@yourtenant.com"  # <-- Replace with your username
        password = "your-password"  # <-- Replace with your password
        try:
            ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
            # Example: get top 5 documents from 'Documents' library
            items = ctx.web.lists.get_by_title("Documents").items.top(5).get().execute_query()
            doc_titles = [item.properties.get("FileLeafRef", "NoName") for item in items]
            return f"📄 Top 5 SharePoint documents: {doc_titles}", ""
        except Exception as e:
            return f"❌ Error retrieving SharePoint documents: {e}", ""
    elif agent_id == "BingDataRetrievalAgent":
        # TODO: Implement Bing/Web search logic or remove if not used
        return f"🌐 Bing search results for: {user_input}", "mock_thread_id"
    else:
        return f"❌ Unknown agent: {agent_id}", ""
