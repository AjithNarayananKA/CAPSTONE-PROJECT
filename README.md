# AI-Based Interview Question Generator

This project uses the Flan T5 model to generate AI-based interview questions. Follow the steps below to set up the environment, install dependencies, fine-tune the model, and run the application.

### Prerequisites

* Python 3.x installed
* Access to Google Colab (recommended for model fine-tuning)
  
### Setup Guide
Step 1: Create a Virtual Environment
* Open a terminal or command prompt.
* Run the following command to create a virtual environment:
  
  * python -m venv myenv
    
Step 2: Activate the Virtual Environment

* On Windows:
  * myenv\Scripts\activate
  
 * On macOS/Linux:
   * source myenv/bin/activate
  
Step 3: Install Required Libraries

* Ensure that requirements.txt is in the project directory.
* Run the following command to install the dependencies:

  * pip install -r requirements.txt
   
Step 4: Download SpaCy Model

* To use the SpaCy library, download the en_core_web_sm language model:


  * python -m spacy download en_core_web_sm
    
Step 5: Fine-Tune the Flan T5 Model

* Open FlanT5.ipynb in Google Colab.
* Follow the notebook instructions to fine-tune the model.
* Save the fine-tuned model in a folder named models.
  
Step 6: Run the Application
* After fine-tuning, run the app.py file to start the application:

    * python app.py
      
### Notes

Google Colab is recommended for the fine-tuning step, as it requires GPU support for optimal performance.
Ensure that the models folder is in the same directory as app.py after fine-tuning the model.
