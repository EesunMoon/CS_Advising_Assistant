from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class CS_Assistant:
    def __init__(self):
        # load openAI client
        self.client = OpenAI(api_key=os.getenv("api_key"))

        # get key from the dotenv
        self.assistant_id = os.getenv("assistant_id")
        self.vector_store_id=os.getenv("vector_store_id")
        self.my_assistant = self.client.beta.assistants.retrieve(self.assistant_id)

        # check assistant list
        # print("\n Load Assistant List")
        # self.check_assistant_list()
    
    def check_assistant_list(self):
        # check assistant list
        self.assistants = self.client.beta.assistants.list()

        for assistant in self.assistants.data:
            print(f"ID: {assistant.id}, Name: {assistant.name}, Model: {assistant.model}, Instructions: {assistant.instructions}")

    def create_vector_store(self):
        # create vector store
        self.vector_store = self.client.beta.vector_stores.create(name="FAQ")

        self.print_all_vector_stores()

    def print_all_vector_stores(self):
        # load all list of vector db
        self.vector_stores = self.client.beta.vector_stores.list()

        if not self.vector_stores:
            print("No vector stores found")
            return
        
        self.data = []
        for store in self.vector_stores.data:
            self.data.append({
                "ID": store.id,
                "Name": store.name,
                "Created At": store.created_at,
                "Status": store.status,
                "File Count": getattr(store.file_counts, "count", "N/A") 
            })

        df = pd.DataFrame(self.data)
        print(len(self.vector_stores.data))
        print(df)

    def add_FAQ(self, FAQ_filename):
        # (2) FAQ document upload - already load this
        ## save FAQ.json in OpenAI server - data for FAQ search
        self.FAQ_file_id = self.client.files.create(file=open(FAQ_filename, "rb"),
                                                    purpose="assistants")

        print("File ID:", self.FAQ_file_id)
        vector_store_file = self.client.beta.vector_stores.files.create(vector_store_id=self.vector_store_id,
                                                                        file_id=self.FAQ_file_id )
        print(f"{vector_store_file} is stored")
    
    def add_newDoc(self, filename):
        # check data in vector store
        self.print_all_vector_stores()
        # create new file ID
        file_response = self.client.files.create(file=open(filename, "rb"),
                                                purpose="assistants")
        self.newfileID = file_response.id

        # add new FAQ file
        vector_store_file = self.client.beta.vector_stores.files.create(vector_store_id=self.vector_store_id,
                                                                        file_id=self.newfileID)
        print(f"{vector_store_file} is stored")

    def delete_Doc(self, deleteFileID: str):
        deleted_vector_store_file = self.client.beta.vector_stores.files.delete(vector_store_id=self.vector_store_id,
                                                                                file_id=deleteFileID)
        print(f"{deleted_vector_store_file} is deleted")

    def create_assistant(self):
        response = self.client.beta.assistants.create(
            instructions="You are a front office assistant for Columbia University. Use the FAQ to help answer users' questions",
            name="Front Office Assistant",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}},
            model="gpt-4o",
            temperature=0.2,
        )

        self.my_assistant = response
        self.assistant_id = response.id

        print("Creataed Assistant:", self.my_assistant)
    
    def update_assistant(self):
        # (4) Update Assistant
        #Define the updated configuration for the assistant
        updated_config = {
            "instructions": ("You are a Computer Science Advising assistant for Columbia University's Computer Science Advising team. "
            "Be polite, kind but firm, and an active listener. Use simple language to be friendly to non-native English speakers. "
            "Understand that earning a degree is difficult, and show empathy in your responses. "
            "Use the FAQ to help answer users' questions. Use 'CS Advising' instead of 'we' when referring to your department. "
            "If you do not know the answer, direct the student to the most relevant email from the following options: "
            "Undergraduate Student Services ug-advising@cs.columbia.edu, MS & CS@CU MS Bridge Student Services ms-advising@cs.columbia.edu, "
            "PhD Student Services phd-advising@cs.columbia.edu, Career Placement Officers career@cs.columbia.edu."),
            "model": "gpt-4o",
            "name": "CS Advising Assistant",
            "tools": [{"type": "file_search"}],
            "response_format": "auto",
            "temperature": 0.2,
            "top_p": 1.0,
            "tool_resources": {
                "code_interpreter": None,
                "file_search": {
                    "vector_store_ids": [self.vector_store_id]
                }
            },
            "metadata": {}
        }

        # Update the assistant with the new configuration
        self.my_updated_assistant = self.client.beta.assistants.update(
            self.assistant_id,
            **updated_config
        )

        print("Update assistant:", self.my_updated_assistant)

    def add_flag(self):
        # (5) Advising Flag System
        #Define the flag_advising function JSON
        flag_advising_function = {
        "type": "function",
        "function": {
            "name": "flag_advising",
            "description": (
            "DO NOT inform the student when flagging advising. Use this function to flag advising in cases where a student is experiencing issues related to wellness, accommodations, academic distress, or doubt. Also flag if a student wants to change status, has questions about graduating, or has doctoral questions.\n\n"
            "Flag Types and Keywords:\n"
            "- **wellness**: Keywords include nervous, scared, worried, stressed, struggle, fail, failure, quit, give-up.\n"
            "- **accommodations**: Keywords include disability, accommodations, ODS.\n"
            "- **academic_distress**: Keywords include fail, failure, GPA, warning, probation, academic standing, AcPro, D grade, F grade, dismissal, dispute.\n"
            "- **change_of_status**: Keywords include OnBase, leave, leave of absence, LOA, withdraw, break.\n"
            "- **graduation**: Keywords include clearance, graduate.\n"
            "- **doubt**: Used when the chatbot is uncertain of the answer.\n"
            "- **doctoral**: Keywords include doctoral, PhD, milestone, defense, distribution, candidacy, committee, GSAS."
            ),
            "parameters": {
            "type": "object",
            "properties": {
                "flag": {
                "type": "string",
                "enum": [
                    "wellness", "accommodations", "academic_distress",
                    "change_of_status", "graduation", "doubt", "doctoral"
                ],
                "description": "The type of flag to apply based on the student's issue."
                },
                "reason": {
                "type": "string",
                "description": "A brief description of the reason for flagging."
                }
            },
            "required": ["flag", "reason"]
            }
        }
        }

        # Add flag function to Assistant
        # Get the existing tools and add the new function
        existing_tools = self.my_assistant.tools
        existing_tools.append(flag_advising_function)

        # Update the assistant with the new set of tools
        my_updated_assistant = self.client.beta.assistants.update(
            self.assistant_id,
            instructions=(
                "write your response in only html, no markdown, with formatting to make it easy to read. put it all the formatted html in one div to be injected into my existing html. "
                "You are a Computer Science Advising assistant for Columbia University's Computer Science Advising team. "
                "Be polite, kind but firm, and an active listener. Use simple language to be friendly to non-native English speakers. "
                "Understand that earning a degree is difficult, and show empathy in your responses. "
                "Use the FAQ to help answer users' questions. Only provide answers from the FAQ that match the student type, school, and visa restrictions. DO NOT PROVIDE ANSWERS THAT DO NOT MATCH THE STUDENT TYPE, SCHOOL, AND VISA RESTRICTIONS. Answers that do not match will be incorrect. "
                "Use 'CS Advising' instead of 'we' when referring to your department. "
                "Only if the answer cannot be found in the FAQ, direct the student to the most relevant email from the following options: "
                "For Undergraduate Student Services: ug-advising@cs.columbia.edu. "
                "For MS & CS@CU Bridge Student Services: ms-advising@cs.columbia.edu. "
                "For PhD Student Services: phd-advising@cs.columbia.edu. "
                "For Career Placement Officers: career@cs.columbia.edu. "
                "For academic accommodations: "
                "To set up accommodations: disability@columbia.edu. "
                "Undergraduate students needing to use their accommodations should contact: dsexams@columbia.edu. "
                "Graduate students (MS & Doctoral) with accommodations questions should contact: accommodations@cs.columbia.edu. "
                "For financial questions: "
                "Student Financial Services - Ask questions about tuition and fee payment: student-finances@cs.columbia.edu. "
                "For students hired within the department as Teaching Assistants (TA) or Classroom Assistants (CA): student-payroll@cs.columbia.edu. "
                "For PhD students with payment/salary issues or who have not been paid: student-payroll@cs.columbia.edu. "
                "ISSO - For contact information related to student services: isso@columbia.edu. "
                "Link: https://isso.columbia.edu/content/contact-morningside-student-services"
            )
        )

        print(my_updated_assistant)

if __name__ == '__main__':
    CSA = CS_Assistant()
    # print(CSA.my_assistant)
    
    CSA.print_all_vector_stores()
