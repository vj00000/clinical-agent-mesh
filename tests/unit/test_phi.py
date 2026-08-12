"""PHI redaction.

The hard requirement is not catching identifiers — it is *not* catching clinical
values that look like them. Over-redaction silently destroys the meaning of the
query: a blood pressure of 120/80 mangled into a date leaves the model answering
a different question.
"""

from mesh.guardrails.phi import redact_phi


def test_an_email_address_is_redacted():
    result = redact_phi("contact me at jane.doe@example.com for records")

    assert "jane.doe@example.com" not in result.text
    assert "[EMAIL]" in result.text


def test_a_phone_number_is_redacted():
    result = redact_phi("call 555-123-4567 to confirm")

    assert "555-123-4567" not in result.text
    assert "[PHONE]" in result.text


def test_a_social_security_number_is_redacted():
    result = redact_phi("SSN 123-45-6789 on file")

    assert "123-45-6789" not in result.text
    assert "[SSN]" in result.text


def test_a_medical_record_number_is_redacted():
    result = redact_phi("patient MRN: 00847213 admitted Tuesday")

    assert "00847213" not in result.text
    assert "[MRN]" in result.text


def test_a_full_date_is_redacted():
    result = redact_phi("date of birth 04/11/1962")

    assert "04/11/1962" not in result.text
    assert "[DATE]" in result.text


def test_a_blood_pressure_reading_is_not_mistaken_for_a_date():
    """120/80 has two components; a date has three. Redacting it would change
    the clinical question being asked."""
    result = redact_phi("blood pressure was 120/80 at rest")

    assert "120/80" in result.text
    assert result.found == []


def test_a_dosage_is_left_intact():
    result = redact_phi("started lisinopril 10 mg once daily")

    assert "10 mg" in result.text
    assert result.found == []


def test_a_lab_value_range_is_left_intact():
    result = redact_phi("HbA1c improved from 8.2 to 6.9 percent")

    assert "8.2" in result.text and "6.9" in result.text
    assert result.found == []


def test_the_categories_found_are_reported():
    result = redact_phi("jane@example.com and 555-123-4567")

    assert set(result.found) == {"EMAIL", "PHONE"}


def test_clean_clinical_text_passes_through_unchanged():
    text = "what is first-line therapy for stage 2 hypertension?"

    result = redact_phi(text)

    assert result.text == text
    assert result.found == []


def test_redaction_is_idempotent():
    """guard_in may run over already-redacted text on a resumed conversation."""
    once = redact_phi("email jane@example.com").text
    twice = redact_phi(once).text

    assert once == twice
