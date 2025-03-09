import streamlit as st
import requests
import os
from PIL import Image

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Set up Streamlit page config
st.set_page_config(page_title="Intel AI Processor Assistant", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Processor Query", "🔧 Troubleshooting Assistant"])

# Initialize session state for indexing status
if "pdf_indexed" not in st.session_state:
    st.session_state["pdf_indexed"] = False

#**Home Page**
if page == "🏠 Home":
    st.title("🚀 AI-Powered Intel Xeon 6 Assistant")
    st.write("""
    Welcome to the **AI-Powered Intel Xeon 6 Optimization & Troubleshooting Assistant**!  
    - **🔍 Processor Query** → Ask AI about Intel Xeon 6.
    - **🔧 Troubleshooting Assistant** → Diagnose and fix Intel product issues with AI.
    """)
    
    # ✅ Load image 
    image_path = "/Users/vemana/Documents/Intel_AI_Processor/frontend/Intel.png"

    if os.path.exists(image_path):
        img1 = Image.open(image_path)

        # ✅ Resize while maintaining aspect ratio (Optional)
        max_width = 450  # Adjust width as needed
        aspect_ratio = img1.height / img1.width
        new_height = int(max_width * aspect_ratio)  # Calculate height automatically

        img1 = img1.resize((max_width, new_height))

        # ✅ Display optimized image in Streamlit
        st.image(img1, caption="Intel Xeon 6", use_column_width=False)  # Keeps original aspect ratio
    else:
        st.warning("⚠️ Image not found: Intel.png")

    # Train LLM Button
    if st.button("🛠 Train AI on Intel Xeon 6 Document"):
        with st.spinner("🔄 Training AI... Please wait."):
            response = requests.post(f"{BACKEND_URL}/train_llm")
            response_data = response.json()

        print(f"Train LLM Response: {response_data}")  # Debugging log

        if response.status_code == 200 and "message" in response_data:
            st.success(response_data["message"])
            st.session_state["pdf_indexed"] = True  # ✅ Store index status
        else:
            st.error(f"❌ Training failed: {response_data.get('error', 'Unknown error')}")

    # Show success message only if the PDF is indexed
    if st.session_state["pdf_indexed"]:
        st.success("✅ PDF is already indexed and ready for queries.")

#**Processor Query**
elif page == "🔍 Processor Query":
    st.title("🔍 Ask AI About Intel Xeon 6 Processors")

    if not st.session_state["pdf_indexed"]:
        st.warning("⚠ Please train the AI first by clicking the button on the Home page!")

    query = st.text_area("💡 Enter your question:")

    if st.button("🤖 Ask AI"):
        if query:
            with st.spinner("🔄 Retrieving AI response... Please wait."):
                response = requests.post(f"{BACKEND_URL}/query_document", json={"query": query})

            if response.status_code == 200:
                ai_response = response.json().get("response")
                print(f"Streamlit AI Response: {ai_response}")  # Debugging

                if ai_response:
                    st.success("### ✅ AI Response:")
                    st.write(ai_response)
                else:
                    st.warning("⚠ No relevant answer found. Try rephrasing your question.")
            else:
                st.error("❌ Error retrieving response. Please try again later.")
        else:
            st.warning("⚠ Please enter a question before asking.")

# 🔧 **Troubleshooting Assistant**
elif page == "🔧 Troubleshooting Assistant":
    st.title("🔧 Intel Product Troubleshooting Assistant")  # ✅ Covers all Intel products

    st.write("""
    Describe your issue with any **Intel product** (e.g., Xeon processors, Core processors, GPUs, NUCs, etc.).
    The AI will retrieve troubleshooting steps from **reliable web sources**.
    """)

    issue = st.text_area("🔍 Describe the issue you're facing:")

    if st.button("🔎 Search for a Solution"):
        if issue:
            with st.spinner("🔄 Searching for troubleshooting steps... (May take up to few seconds)"):
                try:
                    response = requests.post(f"{BACKEND_URL}/analyze", json={"issue": issue}, timeout=60)

                    if response.status_code == 200:
                        ai_solution = response.json().get("solution")

                        if ai_solution:
                            st.success("### ✅ AI-Generated Solution:")
                            st.markdown(ai_solution)  # ✅ Display solution in markdown for better readability
                        else:
                            st.warning("⚠ No solution found. Try rephrasing the issue or checking Intel’s support pages.")
                    else:
                        error_msg = response.json().get('error', 'Unknown error')
                        st.error(f"❌ API Error: {error_msg}")
                except requests.exceptions.Timeout:
                    st.error("⏳ Request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")
        else:
            st.warning("⚠ Please enter an issue before analyzing.")
