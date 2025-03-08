import logging
from pathlib import Path
from data.expand_dataset import DatasetExpander

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_expansion():
    """Test the dataset expansion functionality."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'processed' / 'sample_data.json'
    output_file = base_dir / 'data' / 'processed' / 'expanded_dataset.json'
    
    # Initialize expander with a small target size for testing
    expander = DatasetExpander(
        input_file=str(input_file),
        output_file=str(output_file),
        target_size=100  # Start with 100 examples for testing
    )
    
    # Process data
    expander.process_data()
    
    # Get statistics
    stats = expander.get_statistics()
    
    # Print results
    logger.info("\nTest Results:")
    logger.info(f"Total examples processed: {stats['total_examples']}")
    logger.info(f"Unique subjects found: {stats['unique_subjects']}")
    
    # Verify output file exists
    if output_file.exists():
        logger.info(f"Output file created successfully at {output_file}")
    else:
        logger.error("Output file was not created!")

if __name__ == "__main__":
    test_dataset_expansion() 