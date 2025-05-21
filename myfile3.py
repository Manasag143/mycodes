import requests
import time

# Define the Llama URL - replace with your actual endpoint
LLAMA_URL = "https://your-llama-endpoint.com/llm/"

class LlamaQA:
    """
    A simple class that uses Llama to answer questions based on a provided summary.
    """
    def __init__(self, summary, llama_url=LLAMA_URL):
        """
        Initialize with a summary text that will be used as context for answering questions.
        
        Args:
            summary: The text summary that will be used as context
            llama_url: URL endpoint for the Llama model
        """
        self.summary = summary
        self.llama_url = llama_url
    
    def ask(self, question):
        """
        Ask a question and get an answer based on the summary.
        
        Args:
            question: The question to answer
            
        Returns:
            A string containing the answer
        """
        # Format the prompt with system instruction, summary context, and question
        prompt = self._format_prompt(question)
        
        # Send request to Llama
        response = self._call_llama(prompt)
        
        return response
    
    def _format_prompt(self, question):
        """
        Format the prompt for Llama with the summary as context.
        
        Args:
            question: The question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant. Answer questions based only on the provided context.
If the context doesn't contain the information needed, simply say you don't have
enough information to answer.

Context:
{self.summary}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    def _call_llama(self, prompt):
        """
        Call the Llama API with the formatted prompt.
        
        Args:
            prompt: The formatted prompt string
            
        Returns:
            The response from Llama
        """
        try:
            # Create request body
            body = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            
            # Send POST request to Llama API
            response = requests.post(self.llama_url, json=body, verify=False)
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse the response
                result = response.json()
                return result[0]['generated_text']
            else:
                return f"Error: Received status code {response.status_code}"
                
        except Exception as e:
            return f"Error calling Llama API: {str(e)}"


def main():
    # Example summary - replace with your own summary text
    my_summary = """
    The company database contains information about various companies, their contacts, 
    addresses, and business areas. Key tables include company_mst (master list of companies), 
    contact_mst (contact information), company_address_mst (company addresses), and various 
    reference tables like business_area_ref, industry_ref, and country_ref. The database 
    tracks information like company names, contact details, billing status, industry sectors, 
    and geographical information.
    """
    
    # Initialize the QA system with our summary
    qa_system = LlamaQA(my_summary)
    
    print("Simple Llama QA System")
    print("======================")
    print("Ask questions about the summary or type 'exit' to quit.")
    
    while True:
        # Get user question
        question = input("\nQuestion: ")
        
        # Check if user wants to exit
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        # Skip empty questions
        if not question.strip():
            continue
            
        print("\nThinking...")
        
        # Get answer from Llama
        start_time = time.time()
        answer = qa_system.ask(question)
        end_time = time.time()
        
        # Print the answer
        print(f"\nAnswer ({end_time - start_time:.2f} seconds):")
        print("---------------------------------------")
        print(answer)
        print("---------------------------------------")


if __name__ == "__main__":
    main()
