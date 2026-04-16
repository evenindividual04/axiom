from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow import Workflow, WorkflowExecutionInput
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

email_classification_agent = Agent(
    name="email classification agent",
    model=OpenAIChat(id="gpt-4o"),
    role="you are a email classification agent, who can classify the email to the given categories.",
    instructions=dedent(f"""\
        Classify the following customer email into exactly one of these categories:
        - Subscription
        - Dissatisfaction  
        - MoneyRefund
        Respond with only the category name, nothing else.\
    """)
)

email_content_extractor_agent = Agent(
    name="email content extractor agent",
    model=OpenAIChat(id="gpt-4o"),
    role="",
    instructions=dedent(f"""
        Scenario-1
        Extract the following information from the given subscription email:
        - transaction_number: The transaction number (e.g., "PAY-2024-001")
        - amount: The total amount mentioned (without $ or ₹ sign, e.g., "299.99")
            
        Respond with valid JSON only, like:
        {{"transaction_number": "PAY-2024-001", "amount": "299.99"}}
        
        Scenario-2:
        Extract the following information from the given subscription email:
        - transaction_number: The transaction number (e.g., "PAY-2024-001")
        - amount: The total amount mentioned (without $ or ₹ sign, e.g., "299.99")
            
        Respond with valid JSON only, like:
        {{"transaction_number": "PAY-2024-001", "amount": "299.99"}}
        
        Scenario-3:
        Extract the following information from the given subscription email:
        - transaction_number: The transaction number (e.g., "PAY-2024-001")
        - amount: The total amount mentioned (without $ or ₹ sign, e.g., "299.99")
            
        Respond with valid JSON only, like:
        {{"transaction_number": "PAY-2024-001", "amount": "299.99"}}
        
        If a field is not found, use null.
        """)
)

email_response_generation_agent = Agent(
    name="email response generation agent",
    model=OpenAIChat(id="gpt-4o"),
    role="",
    instructions=dedent(f"""
        Generate a professional customer support response template for the following:
        
        The response should:
        - Be polite and professional
        - Acknowledge the specific issue type
        - Include next steps or resolution process
        - Reference any extracted information appropriately
        
        Keep it concise but helpful.
        """)
)


def custom_execution_function(workflow: Workflow, execution_input: WorkflowExecutionInput):
    print(f"Executing Workflow: {workflow.name}")
    # Run the email classification agent
    classification_response = email_classification_agent.run(input=execution_input.input)
    content_response = email_content_extractor_agent.run(
        input=f"""\Email content: {execution_input.input}, Email Category: {classification_response.content}""")
    email_response = email_response_generation_agent.run(input=f"""\Extracted Content: {content_response.content}, 
    Email Category: {classification_response.content}, Email Content: {execution_input.input}""")
    # return the response for email
    return email_response.content


email_creation_workflow = Workflow(
    name="Email Creation Workflow",
    description="Automated Email Response creation from context",
    steps=custom_execution_function,
)

# Test workflow standalone
# _resp = content_creation_workflow.run(
#     input="Hello, I paid for the Premium subscription on November 25th using my credit card ending in 7834. The payment of $49.99 was deducted successfully (transaction ID: PAY-20241125-4521), but my account portal still shows 'Free Plan' status. It's been 3 days now and I need access to premium features urgently for my project.,category Subscription Issue; plan_type Premium; payment_date Nov 25th; amount 49.99; transaction_id PAY-20241125-4521; card_last_digits 7834; current_status Free Plan; response confirms payment and activates subscription"
# )
#
# print(_resp.content)
