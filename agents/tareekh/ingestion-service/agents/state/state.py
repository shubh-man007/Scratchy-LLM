from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class State:
    # Input Layer
    bucket_name: str = ""
    folder_path: str = ""  
    local_folder_path: str = ""  
    downloaded_files: Dict[str, str] = field(default_factory=dict)  
    file_hashes: Dict[str, str] = field(default_factory=dict)  
    file_sizes: Dict[str, int] = field(default_factory=dict)  
    mime_types: Dict[str, str] = field(default_factory=dict)  
    detected_types: Dict[str, str] = field(default_factory=dict)  
    gcs_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing Layer
    raw_contents: Dict[str, str] = field(default_factory=dict)  
    chunks: Dict[str, List[Any]] = field(default_factory=dict)  
    extracted_content: Dict[str, Any] = field(default_factory=dict)  
    chunked_documents: Dict[str, List[Any]] = field(default_factory=dict)  
    contract_types: Dict[str, str] = field(default_factory=dict)  
    parties: Dict[str, List[str]] = field(default_factory=dict)  
    key_dates: Dict[str, Dict[str, str]] = field(default_factory=dict)  
    jurisdictions: Dict[str, str] = field(default_factory=dict)  
    
    # Agent Layer
    risk_assessments: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    redlines: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  #
    playbook_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Output Layer
    vector_ids: Dict[str, List[str]] = field(default_factory=dict)  
    summaries: Dict[str, str] = field(default_factory=list)
    processing_log: List[str] = field(default_factory=list)
    
    # Control Flow
    current_step: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_log(self, message: str):
        self.processing_log.append(f"[{self.current_step}] {message}")
    
    def add_error(self, error: str):
        self.errors.append(f"[{self.current_step}] {error}")
    
    def add_warning(self, warning: str):
        self.warnings.append(f"[{self.current_step}] {warning}")
