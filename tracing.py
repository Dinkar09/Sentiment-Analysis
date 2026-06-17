"""
BERT Execution Tracker Module

This module provides functionality to trace and save BERT model final output
(pooled output for classification tasks).
"""

import torch
import json
import os
from datetime import datetime


class BertExecutionTracker:
    """
    Tracker class for BERT final execution output.
    
    Saves the final pooled output representation for analysis and debugging purposes.
    """
    
    def __init__(self, output_dir="bert_execution_outputs"):
        """
        Initialize the tracker.
        
        Args:
            output_dir (str): Directory to save output JSON files
        """
        self.output_dir = output_dir
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created directory: {self.output_dir}")
    
    def _tensor_to_json_serializable(self, tensor_data):
        """
        Convert PyTorch tensor to JSON-serializable format.
        
        Args:
            tensor_data: PyTorch tensor or numpy array
            
        Returns:
            List: JSON-serializable representation of the tensor
        """
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data.detach().cpu().numpy().tolist()
        else:
            return tensor_data
    
    def _save_file(self, data, filename):
        """
        Save data to JSON file with metadata.
        
        Args:
            data: Tensor data to save
            filename (str): Name of the JSON file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        output_json = {
            "timestamp": datetime.now().isoformat(),
            "tensor_shape": str(data.shape) if isinstance(data, torch.Tensor) else "unknown",
            "data": self._tensor_to_json_serializable(data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"✓ Saved: {filename}")
    
    def trace_forward_pass(self, bert_model, input_ids, segment_ids, mask=None):
        """
        Run BERT forward pass and save final pooled output.
        
        Args:
            bert_model: BERT model instance
            input_ids: Token IDs [batch_size, seq_len]
            segment_ids: Segment labels [batch_size, seq_len]
            mask: Attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Tuple: (sequence_output, pooled_output) from BERT
        """
        
        # Run forward pass
        sequence_output, pooled_output = bert_model(input_ids, segment_ids, mask)
        
        # Save final pooled output
        print("\n" + "="*60)
        print("BERT EXECUTION TRACKING - FINAL OUTPUT")
        print("="*60 + "\n")
        
        print("[FINAL OUTPUT] POOLED OUTPUT (CLS Token Representation)")
        print(f"  Shape: {pooled_output.shape}")
        print(f"  Meaning: [Batch_size={pooled_output.shape[0]}, Hidden_dim={pooled_output.shape[1]}]")
        print(f"  Purpose: Used for sentence-level classification tasks")
        self._save_file(pooled_output, "BERT_FINAL_OUTPUT.json")
        
        print("\n" + "="*60)
        print("BERT EXECUTION TRACKING COMPLETED")
        print(f"Output files saved in: {self.output_dir}/")
        print("="*60 + "\n")
        
        return sequence_output, pooled_output