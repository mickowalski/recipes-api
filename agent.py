import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from github import Auth, Github
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_URL")

repo_url = "https://github.com/mickowalski/recipes-api.git"

git = Github(auth=Auth.Token(token)) if token else Github()
full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
repo = git.get_repo(full_repo_name)

context_agent_system_prompt = """
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""

commentor_agent_system_prompt = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Context Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. 
"""

review_and_posting_agent_system_prompt = """
You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  
"""


def get_pr_details(pr_number: int) -> dict:
    """retrieves Pull Request details from repository"""
    commit_shas = []
    pr = repo.get_pull(pr_number)
    commits = pr.get_commits()

    for c in commits:
        commit_shas.append(c.sha)

    pr_details = {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha,
        "commits_SHAs": commit_shas,
    }

    return pr_details


pr_details_tool = FunctionTool.from_defaults(get_pr_details)


def get_changed_files(pr_number: int) -> list[dict[str, Any]]:
    """Retrieve all changed files and their patches from the complete pull request."""
    pr = repo.get_pull(pr_number)
    return [
        {
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "changes": file.changes,
            "patch": file.patch or "",
        }
        for file in pr.get_files()
    ]


changed_files_tool = FunctionTool.from_defaults(get_changed_files)


def get_file_contents(path: str, ref: str | None = None) -> str:
    """Retrieve a file's text by repository-relative path, optionally at a commit or branch."""
    contents = repo.get_contents(path, ref=ref) if ref else repo.get_contents(path)
    if isinstance(contents, list):
        raise ValueError(f"Expected a file, but {path!r} is a directory")
    return contents.decoded_content.decode("utf-8")


file_contents_tool = FunctionTool.from_defaults(get_file_contents)


async def add_gathered_context(ctx: Context, gathered_context: str) -> str:
    """adds gathered context to the state"""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_context
    await ctx.store.set("state", current_state)
    return "State updated with gathered context"


add_gathered_context_tool = FunctionTool.from_defaults(add_gathered_context)


async def add_draft_comment(ctx: Context, draft_comment: str) -> str:
    """adds draft comment to the state"""
    current_state = await ctx.store.get("state")
    current_state["draft_comments"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment"


add_draft_comment_tool = FunctionTool.from_defaults(add_draft_comment)


async def add_final_review(ctx: Context, final_review: str) -> str:
    """adds final review for Pull request to the state"""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review"


add_final_review_tool = FunctionTool.from_defaults(add_final_review)


async def post_final_review(pr_number: int, final_review: str) -> str:
    """posts final review for PR to GitHub"""
    pr = repo.get_pull(pr_number)
    pr.create_review(body=final_review)
    return "Pull request updated with final review"


post_final_review_tool = FunctionTool.from_defaults(post_final_review)

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    api_base=api_base,
)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    tools=[pr_details_tool, changed_files_tool, file_contents_tool, add_gathered_context_tool],
    system_prompt=context_agent_system_prompt,
    description="Gathers all the needed context form GitHub",
    can_handoff_to=["CommentorAgent"]
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment",
    tools=[add_draft_comment_tool],
    system_prompt=commentor_agent_system_prompt,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Takes the draft review created by commentor agent, makes it review and post it to github ",
    tools=[add_final_review_tool, post_final_review_tool],
    system_prompt=review_and_posting_agent_system_prompt,
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if git is not None:
            git.close()
