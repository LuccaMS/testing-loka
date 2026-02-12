Todos os arquivos tem a seguinte estrutura. contendo patient information e clinician information  

## Patient Information
*   **Patient ID:** p7a8b9c0-d1e2-3f4a-5b6c-7d8e9f0a1b2c
*   **Name:** Sarah Johnson
*   **Date of Birth:** March 15, 1985
*   **Gender:** Female
*   **Address:** 123 Maple Street, Anytown, CA 90210
*   **Contact Number:** (555) 123-4567

Or it can be like that 

## Patient Information

*   **Patient ID**: PT-987654
*   **Name**: Erica Jane Smith
*   **Date of Birth**: November 15, 1978 (45 years old)
*   **Gender**: Female
*   **Address**: 456 Oak Avenue, Springfield, IL 62704
*   **Contact Number**: 555-901-2345

Notice that the difference is that the ":" is after "**" and in another its before.

## Clinician Information
*   **Clinician ID:** c1d2e3f4-a5b6-7c8d-9e0f-1a2b3c4d5e6f
*   **Name:** Dr. Emily White
*   **Specialization:** Internal Medicine
*   **Institution:** City General Hospital
*   **Contact Email:** emily.white@hospital.com

The same goes for clinician 

## Clinician Information

*   **Clinician ID**: CLI-5678
*   **Name**: Dr. Benjamin Carter
*   **Specialization**: Internal Medicine
*   **Institution**: City General Hospital
*   **Contact Email**: b.carter@citygen.org


All the files have the field named  "Chief Complaint", but i have detected three patterns.
Some are like this "---

## Chief Complaint

Acute abdominal pain, nausea, and vomiting for 24 hours.

---
" they start after a --- have a ## and the text, and after that --- again.

In some others its like this "**Chief Complaint:**" and then the value until another *.
For other type, there is a field called "## Clinical Details" that works like this "## Clinical Details

*   **Chief Complaint:** Routine check-up, no specific complaints.
*   **History of Present Illness:** Not provided.
*   **Past Medical History:** No significant past medical history.
"

For the field of "History of Present Illness" we have two options. 


