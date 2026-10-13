import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Schema
class Resume(BaseModel):
    name: str = Field(description="Candidate ka poora naam")
    email: str = Field(description="Email address")
    skills: list[str] = Field(description="Technical skills ki list")
    experience_years: int = Field(description="Total years of experience")
    is_hireable: bool = Field(description="Kya candidate hire karne layak hai")

# 2. Parser
parser = PydanticOutputParser(pydantic_object=Resume)

# 3. Prompt
Prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured info from resume. Return format:\n{format_instructions}"),
    ("human", "Resume text:\n{resume_text}")
]).partial(format_instructions=parser.get_format_instructions())

# 4. Env
load_dotenv()

# 5. LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model="openrouter/free",
    temperature=0.0,
)

# 6. Chain
chain = Prompt | llm | parser

# 7. Input
resume_text = """
Hi, I'm Shobhit Sharma. I have 5 years of experience in Python, 
Django, and AWS. You can reach me at shobhit@example.com.
I've built 10+ production APIs and led a team of 3 developers.
"""

# 8. Invoke
result = chain.invoke({"resume_text": resume_text})

print(type(result))
print(result.name)
print(result.email)
print(result.skills)
print(result.experience_years)
print(result.is_hireable)