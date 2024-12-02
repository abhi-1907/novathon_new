import os
import pandas as pd
import logging
import llmware
from llmware.models import FinetuningModel
from llmware.dataset_tools import Dataset
import llmware.prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class BERTFineTuner:
    def __init__(self, dataset_path, base_model='bert-base-uncased'):
        """
        Initialize the BERT Fine-tuning process
        
        Args:
            dataset_path (str): Path to the CSV dataset
            base_model (str): Base BERT model to fine-tune
        """
        self.dataset_path = dataset_path
        self.base_model = base_model
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    def load_and_prepare_dataset(self, sample_size=None):
        """
        Load and prepare dataset for fine-tuning
        
        Args:
            sample_size (int, optional): Number of rows to sample for testing
        
        Returns:
            list: Prepared dataset for fine-tuning
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.dataset_path)
            
            # Optional sampling for testing
            if sample_size:
                df = df.sample(min(sample_size, len(df)))
            
            # Prepare data for LLMWare
            prepared_data = []
            
            for _, row in df.iterrows():
                # Combine relevant fields into a single text
                text = f"Name: {row.get('Name', 'Unknown')}, " \
                       f"Age: {row.get('Age', 'Unknown')}, " \
                       f"Gender: {row.get('Gender', 'Unknown')}, " \
                       f"Medical Condition: {row.get('Medical Condition', 'Unknown')}, " \
                       f"Blood Type: {row.get('Blood Type', 'Unknown')}"
                
                # Choose label - can be modified based on your specific use case
                label = str(row.get('Medical Condition', 'Unknown'))
                
                prepared_data.append({
                    "text": text,
                    "label": label
                })
            
            logger.info(f"Prepared {len(prepared_data)} data points for fine-tuning")
            return prepared_data
        
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def create_llmware_dataset(self, prepared_data):
        """
        Create a custom dataset in LLMWare
        
        Args:
            prepared_data (list): Prepared dataset
        
        Returns:
            Dataset: LLMWare dataset
        """
        try:
            custom_dataset = Dataset().create_new_dataset("healthcare_dataset")
            custom_dataset.add_huggingface_dataset(prepared_data)
            
            logger.info("Successfully created LLMWare dataset")
            return custom_dataset
        
        except Exception as e:
            logger.error(f"Error creating LLMWare dataset: {e}")
            raise
    
    def finetune_model(self, custom_dataset, output_model_name='finetuned_healthcare_bert'):
        """
        Fine-tune BERT model using LLMWare
        
        Args:
            custom_dataset (Dataset): Prepared LLMWare dataset
            output_model_name (str): Name for the saved fine-tuned model
        
        Returns:
            FinetuningModel: Fine-tuned model
        """
        try:
            # Initialize the Finetuning Model
            model = FinetuningModel()
            
            # Configure fine-tuning parameters
            model.set_finetuning_params(
                model_name=self.base_model,
                batch_size=16,
                epochs=3,
                learning_rate=2e-5
            )
            
            # Perform fine-tuning
            logger.info("Starting model fine-tuning...")
            model.finetune(
                dataset=custom_dataset, 
                task_type="classification"
            )
            
            # Save the fine-tuned model
            output_path = model.save_model(output_model_name)
            logger.info(f"Model saved successfully at: {output_path}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error during model fine-tuning: {e}")
            raise
    
    def test_model(self, model, test_prompts=None):
        """
        Test the fine-tuned model with sample prompts
        
        Args:
            model (FinetuningModel): Fine-tuned model
            test_prompts (list, optional): List of test prompts
        
        Returns:
            dict: Inference results
        """
        if not test_prompts:
            test_prompts = [
                "Name: John Doe, Age: 45, Gender: Male, Medical Condition: Diabetes",
                "Name: Jane Smith, Age: 35, Gender: Female, Medical Condition: Hypertension"
            ]
        
        results = {}
        for prompt in test_prompts:
            try:
                response = model.inference(prompt)
                results[prompt] = response
                logger.info(f"Prompt: {prompt}\nPrediction: {response}")
            except Exception as e:
                logger.error(f"Inference error for prompt '{prompt}': {e}")
        
        return results

def main():
    """
    Main execution function
    """
    try:
        # Replace with your actual dataset path
        dataset_path = 'records/healthcare_dataset/healthcare_dataset.csv'
        
        # Initialize fine-tuner
        fine_tuner = BERTFineTuner(dataset_path)
        
        # Prepare dataset (sample 500 rows for testing)
        prepared_data = fine_tuner.load_and_prepare_dataset(sample_size=500)
        
        # Create LLMWare dataset
        custom_dataset = fine_tuner.create_llmware_dataset(prepared_data)
        
        # Fine-tune the model
        finetuned_model = fine_tuner.finetune_model(
            custom_dataset, 
            output_model_name='healthcare_bert_model'
        )
        
        # Test the model
        fine_tuner.test_model(finetuned_model)
    
    except Exception as e:
        logger.error(f"Fine-tuning process failed: {e}")

if __name__ == "__main__":
    main()