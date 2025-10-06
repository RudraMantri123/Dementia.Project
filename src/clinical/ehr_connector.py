"""EHR connectivity module with FHIR standard support."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
import json

from src.database.models import ClinicalData, UserProfile
from src.database.connection import get_db_manager


class FHIRConnector:
    """FHIR-compliant EHR connector for clinical data integration."""

    def __init__(self, fhir_base_url: str = None, api_key: str = None):
        """
        Initialize FHIR connector.

        Args:
            fhir_base_url: Base URL of FHIR server
            api_key: API key for authentication
        """
        self.base_url = fhir_base_url
        self.api_key = api_key
        self.db_manager = get_db_manager()
        self.headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }

        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def fetch_patient_data(self, fhir_patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch patient data from FHIR server.

        Args:
            fhir_patient_id: FHIR patient identifier

        Returns:
            Patient resource or None
        """
        if not self.base_url:
            return self._get_mock_patient_data(fhir_patient_id)

        try:
            url = f"{self.base_url}/Patient/{fhir_patient_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching patient data: {e}")
            return None

    def fetch_observations(
        self,
        fhir_patient_id: str,
        observation_codes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch clinical observations for patient.

        Args:
            fhir_patient_id: FHIR patient identifier
            observation_codes: List of LOINC codes to filter

        Returns:
            List of Observation resources
        """
        if not self.base_url:
            return self._get_mock_observations(fhir_patient_id)

        try:
            url = f"{self.base_url}/Observation"
            params = {'patient': fhir_patient_id}

            if observation_codes:
                params['code'] = ','.join(observation_codes)

            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()

            bundle = response.json()
            return bundle.get('entry', [])

        except requests.exceptions.RequestException as e:
            print(f"Error fetching observations: {e}")
            return []

    def fetch_conditions(self, fhir_patient_id: str) -> List[Dict[str, Any]]:
        """
        Fetch patient conditions/diagnoses.

        Args:
            fhir_patient_id: FHIR patient identifier

        Returns:
            List of Condition resources
        """
        if not self.base_url:
            return self._get_mock_conditions(fhir_patient_id)

        try:
            url = f"{self.base_url}/Condition"
            params = {'patient': fhir_patient_id}

            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()

            bundle = response.json()
            return bundle.get('entry', [])

        except requests.exceptions.RequestException as e:
            print(f"Error fetching conditions: {e}")
            return []

    def fetch_medications(self, fhir_patient_id: str) -> List[Dict[str, Any]]:
        """
        Fetch patient medication statements.

        Args:
            fhir_patient_id: FHIR patient identifier

        Returns:
            List of MedicationStatement resources
        """
        if not self.base_url:
            return self._get_mock_medications(fhir_patient_id)

        try:
            url = f"{self.base_url}/MedicationStatement"
            params = {'patient': fhir_patient_id}

            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()

            bundle = response.json()
            return bundle.get('entry', [])

        except requests.exceptions.RequestException as e:
            print(f"Error fetching medications: {e}")
            return []

    def sync_patient_to_db(self, user_id: str, fhir_patient_id: str) -> Dict[str, Any]:
        """
        Sync patient data from EHR to local database.

        Args:
            user_id: Internal user identifier
            fhir_patient_id: FHIR patient identifier

        Returns:
            Sync status
        """
        with self.db_manager.get_session() as session:
            # Fetch data from EHR
            patient_data = self.fetch_patient_data(fhir_patient_id)
            conditions = self.fetch_conditions(fhir_patient_id)
            medications = self.fetch_medications(fhir_patient_id)
            observations = self.fetch_observations(fhir_patient_id)

            # Update user profile
            profile = session.query(UserProfile).filter_by(user_id=user_id).first()

            if not profile:
                return {'error': 'User profile not found'}

            # Extract and update profile data
            if patient_data:
                profile.name = self._extract_patient_name(patient_data)
                profile.age = self._calculate_age(patient_data)

            # Store conditions
            for condition_entry in conditions:
                condition = condition_entry.get('resource', {})
                self._store_condition(session, user_id, fhir_patient_id, condition)

            # Store medications
            medications_list = []
            for med_entry in medications:
                med = med_entry.get('resource', {})
                medications_list.append(self._extract_medication_info(med))

            profile.medications = medications_list

            # Store cognitive assessments from observations
            for obs_entry in observations:
                obs = obs_entry.get('resource', {})
                if self._is_cognitive_assessment(obs):
                    self._store_assessment(session, user_id, fhir_patient_id, obs)

            session.commit()

            return {
                'status': 'success',
                'user_id': user_id,
                'fhir_patient_id': fhir_patient_id,
                'synced_at': datetime.utcnow().isoformat(),
                'data_summary': {
                    'conditions': len(conditions),
                    'medications': len(medications),
                    'observations': len(observations)
                }
            }

    def create_observation(
        self,
        fhir_patient_id: str,
        observation_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create new observation in EHR system.

        Args:
            fhir_patient_id: FHIR patient identifier
            observation_data: Observation data

        Returns:
            Created Observation resource or None
        """
        if not self.base_url:
            return {'status': 'mock_created', 'id': 'mock-obs-123'}

        try:
            url = f"{self.base_url}/Observation"

            # Build FHIR Observation resource
            observation = self._build_observation_resource(fhir_patient_id, observation_data)

            response = requests.post(url, headers=self.headers, json=observation, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error creating observation: {e}")
            return None

    def _extract_patient_name(self, patient_data: Dict[str, Any]) -> str:
        """Extract patient name from FHIR resource."""
        names = patient_data.get('name', [])
        if names:
            name = names[0]
            given = ' '.join(name.get('given', []))
            family = name.get('family', '')
            return f"{given} {family}".strip()
        return "Unknown"

    def _calculate_age(self, patient_data: Dict[str, Any]) -> Optional[int]:
        """Calculate age from birth date."""
        birth_date = patient_data.get('birthDate')
        if birth_date:
            try:
                birth = datetime.fromisoformat(birth_date)
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                return age
            except:
                pass
        return None

    def _store_condition(
        self,
        session,
        user_id: str,
        fhir_patient_id: str,
        condition: Dict[str, Any]
    ):
        """Store condition in database."""
        clinical_data = ClinicalData(
            user_id=user_id,
            fhir_patient_id=fhir_patient_id,
            assessment_type='condition',
            diagnosis_code=self._extract_code(condition),
            diagnosis_description=self._extract_display(condition),
            diagnosis_date=self._extract_date(condition, 'onsetDateTime'),
            assessment_data=condition
        )
        session.add(clinical_data)

    def _store_assessment(
        self,
        session,
        user_id: str,
        fhir_patient_id: str,
        observation: Dict[str, Any]
    ):
        """Store cognitive assessment in database."""
        clinical_data = ClinicalData(
            user_id=user_id,
            fhir_patient_id=fhir_patient_id,
            assessment_type=self._extract_assessment_type(observation),
            assessment_score=self._extract_value(observation),
            assessment_date=self._extract_date(observation, 'effectiveDateTime'),
            assessment_data=observation
        )
        session.add(clinical_data)

    def _extract_medication_info(self, medication: Dict[str, Any]) -> Dict[str, str]:
        """Extract medication information."""
        return {
            'name': self._extract_display(medication),
            'code': self._extract_code(medication),
            'status': medication.get('status', 'unknown')
        }

    def _is_cognitive_assessment(self, observation: Dict[str, Any]) -> bool:
        """Check if observation is a cognitive assessment."""
        code = self._extract_code(observation)
        # Common cognitive assessment LOINC codes
        cognitive_codes = ['72172-1', '72173-9', '72107-7']  # MMSE, MoCA, etc.
        return code in cognitive_codes

    def _extract_code(self, resource: Dict[str, Any]) -> Optional[str]:
        """Extract primary code from resource."""
        code_concept = resource.get('code', {})
        codings = code_concept.get('coding', [])
        if codings:
            return codings[0].get('code')
        return None

    def _extract_display(self, resource: Dict[str, Any]) -> str:
        """Extract display text from resource."""
        code_concept = resource.get('code', {}) or resource.get('medicationCodeableConcept', {})
        codings = code_concept.get('coding', [])
        if codings:
            return codings[0].get('display', 'Unknown')
        return code_concept.get('text', 'Unknown')

    def _extract_value(self, observation: Dict[str, Any]) -> Optional[float]:
        """Extract numeric value from observation."""
        value = observation.get('valueQuantity', {})
        return value.get('value')

    def _extract_date(self, resource: Dict[str, Any], field: str) -> Optional[datetime]:
        """Extract and parse date from resource."""
        date_str = resource.get(field)
        if date_str:
            try:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                pass
        return None

    def _extract_assessment_type(self, observation: Dict[str, Any]) -> str:
        """Extract assessment type from observation."""
        display = self._extract_display(observation)
        if 'MMSE' in display:
            return 'MMSE'
        elif 'MoCA' in display:
            return 'MoCA'
        elif 'CDR' in display:
            return 'CDR'
        return 'cognitive_assessment'

    def _build_observation_resource(
        self,
        fhir_patient_id: str,
        observation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build FHIR Observation resource."""
        return {
            'resourceType': 'Observation',
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                    'code': 'cognitive',
                    'display': 'Cognitive Assessment'
                }]
            }],
            'code': observation_data.get('code'),
            'subject': {
                'reference': f'Patient/{fhir_patient_id}'
            },
            'effectiveDateTime': observation_data.get('date', datetime.utcnow().isoformat()),
            'valueQuantity': observation_data.get('value'),
            'note': observation_data.get('notes', [])
        }

    # Mock data methods for testing without actual EHR
    def _get_mock_patient_data(self, fhir_patient_id: str) -> Dict[str, Any]:
        """Get mock patient data for testing."""
        return {
            'resourceType': 'Patient',
            'id': fhir_patient_id,
            'name': [{'given': ['John'], 'family': 'Doe'}],
            'birthDate': '1940-01-01',
            'gender': 'male'
        }

    def _get_mock_observations(self, fhir_patient_id: str) -> List[Dict[str, Any]]:
        """Get mock observations for testing."""
        return [{
            'resource': {
                'resourceType': 'Observation',
                'code': {
                    'coding': [{
                        'system': 'http://loinc.org',
                        'code': '72172-1',
                        'display': 'MMSE Score'
                    }]
                },
                'valueQuantity': {'value': 24, 'unit': 'points'},
                'effectiveDateTime': '2024-01-01T00:00:00Z'
            }
        }]

    def _get_mock_conditions(self, fhir_patient_id: str) -> List[Dict[str, Any]]:
        """Get mock conditions for testing."""
        return [{
            'resource': {
                'resourceType': 'Condition',
                'code': {
                    'coding': [{
                        'system': 'http://snomed.info/sct',
                        'code': '26929004',
                        'display': "Alzheimer's disease"
                    }]
                },
                'onsetDateTime': '2020-01-01T00:00:00Z'
            }
        }]

    def _get_mock_medications(self, fhir_patient_id: str) -> List[Dict[str, Any]]:
        """Get mock medications for testing."""
        return [{
            'resource': {
                'resourceType': 'MedicationStatement',
                'medicationCodeableConcept': {
                    'coding': [{
                        'system': 'http://www.nlm.nih.gov/research/umls/rxnorm',
                        'code': '352447',
                        'display': 'Donepezil'
                    }]
                },
                'status': 'active'
            }
        }]
