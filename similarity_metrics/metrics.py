import torch

def cosine_similarity(image1, image2):
    """
    Calculate the Cosine Similarity score between two batches of images.
    
    Args:
        image1 (torch.Tensor): First image tensor of shape (B, H, W).
        image2 (torch.Tensor): Second image tensor of shape (B, H, W).
    
    Returns:
        torch.Tensor: Cosine similarity scores of shape (B,).
    """
    # Flatten images while preserving batch dimension
    image1_flat = image1.reshape(image1.shape[0], -1)
    image2_flat = image2.reshape(image2.shape[0], -1)
    
    # Calculate cosine similarity along the flattened dimension
    cos = torch.nn.CosineSimilarity(dim=1)
    return cos(image1_flat, image2_flat)

def ncc_similarity(image1, image2):
    """
    Calculate the Normalized Cross-Correlation (NCC) similarity score between two batches of images.
    
    Args:
        image1 (torch.Tensor): First image tensor of shape (B, H, W).
        image2 (torch.Tensor): Second image tensor of shape (B, H, W).
    
    Returns:
        torch.Tensor: NCC similarity scores of shape (B,).
    """
    # Flatten images while preserving batch dimension
    image1_flat = image1.reshape(image1.shape[0], -1)
    image2_flat = image2.reshape(image2.shape[0], -1)
    
    # Calculate means along the flattened dimension
    image1_mean = torch.mean(image1_flat, dim=1, keepdim=True)
    image2_mean = torch.mean(image2_flat, dim=1, keepdim=True)
    
    # Calculate NCC for each batch
    numerator = torch.sum((image1_flat - image1_mean) * (image2_flat - image2_mean), dim=1)
    denominator = torch.sqrt(
        torch.sum((image1_flat - image1_mean)**2, dim=1) * 
        torch.sum((image2_flat - image2_mean)**2, dim=1)
    )
    
    return numerator / denominator

def histogram_similarity(image1, image2, bins=256):
    """
    Calculate the Histogram Intersection similarity score between two batches of images.
    
    Args:
        image1 (torch.Tensor): First image tensor of shape (B, H, W).
        image2 (torch.Tensor): Second image tensor of shape (B, H, W).
        bins (int): Number of bins for the histogram.
    
    Returns:
        torch.Tensor: Histogram intersection similarity scores of shape (B,).
    """
    batch_size = image1.shape[0]
    scores = []
    
    for i in range(batch_size):
        # Calculate histogram for each image in the batch
        hist1 = torch.histc(image1[i], bins=bins, min=0, max=1)
        hist2 = torch.histc(image2[i], bins=bins, min=0, max=1)
        
        # Calculate intersection
        intersection = torch.min(hist1, hist2).sum()
        score = intersection / min(hist1.sum(), hist2.sum())
        scores.append(score)
    
    return torch.tensor(scores, device=image1.device)