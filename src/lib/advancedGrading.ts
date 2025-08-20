import * as tf from '@tensorflow/tfjs'

// Enhanced grading configuration
export interface GradingConfig {
  thresholds: {
    'A+': number
    'A': number
    'B': number
    'C': number
    'D': number
    'F': number
  }
  styleAdjustments: {
    cartoon: number
    sketch: number
    realistic: number
    abstract: number
  }
}

export const defaultGradingConfig: GradingConfig = {
  thresholds: {
    'A+': 90,
    'A': 80,
    'B': 70,
    'C': 60,
    'D': 50,
    'F': 0
  },
  styleAdjustments: {
    cartoon: 1.1,    // More lenient for cartoon style
    sketch: 1.05,    // Slightly more lenient for sketches
    realistic: 1.0,  // Standard grading
    abstract: 0.95   // Slightly stricter for abstract
  }
}

// Enhanced comparison result
export interface EnhancedComparisonResult {
  similarity: number
  ssimScore: number
  grade: string
  color: string
  style: 'cartoon' | 'sketch' | 'realistic' | 'abstract'
  styleConfidence: number
  feedback: string[]
  heatmapUrl?: string
}

// Calculate multiple similarity metrics
export async function calculateEnhancedSimilarity(
  img1Url: string,
  img2Url: string,
  model: any
): Promise<{
  cosineSimilarity: number
  ssimScore: number
  structuralSimilarity: number
}> {
  const [features1, features2] = await Promise.all([
    extractFeatures(img1Url, model),
    extractFeatures(img2Url, model)
  ])

  // Cosine similarity
  const cosineSim = calculateCosineSimilarity(features1, features2)
  
  // Simplified SSIM approximation
  const ssimScore = await calculateSimplifiedSSIM(img1Url, img2Url)
  
  // Structural similarity based on feature distribution
  const structuralSim = calculateStructuralSimilarity(features1, features2)

  return {
    cosineSimilarity: cosineSim,
    ssimScore: ssimScore,
    structuralSimilarity: structuralSim
  }
}

// Extract features from image
async function extractFeatures(imageUrl: string, model: any): Promise<number[]> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = async () => {
      try {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .expandDims(0)
          .cast('float32')
          .div(255.0)

        const embeddings = model.infer(tensor, true) as tf.Tensor
        const features = Array.from(await embeddings.data())
        
        tensor.dispose()
        embeddings.dispose()
        
        resolve(features)
      } catch (error) {
        reject(error)
      }
    }
    img.onerror = reject
    img.src = imageUrl
  })
}

// Calculate cosine similarity
function calculateCosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0)
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
  return dotProduct / (magnitudeA * magnitudeB)
}

// Simplified SSIM calculation using canvas
async function calculateSimplifiedSSIM(img1Url: string, img2Url: string): Promise<number> {
  return new Promise((resolve) => {
    const canvas1 = document.createElement('canvas')
    const canvas2 = document.createElement('canvas')
    const ctx1 = canvas1.getContext('2d')
    const ctx2 = canvas2.getContext('2d')
    
    if (!ctx1 || !ctx2) {
      resolve(0.5) // Default value if canvas not available
      return
    }

    const img1 = new Image()
    const img2 = new Image()
    let loadedCount = 0

    const processImages = () => {
      loadedCount++
      if (loadedCount === 2) {
        // Resize both images to same size
        const size = 64
        canvas1.width = canvas2.width = size
        canvas1.height = canvas2.height = size
        
        ctx1.drawImage(img1, 0, 0, size, size)
        ctx2.drawImage(img2, 0, 0, size, size)
        
        const data1 = ctx1.getImageData(0, 0, size, size).data
        const data2 = ctx2.getImageData(0, 0, size, size).data
        
        // Calculate simplified structural similarity
        let totalDiff = 0
        for (let i = 0; i < data1.length; i += 4) {
          const r1 = data1[i], g1 = data1[i + 1], b1 = data1[i + 2]
          const r2 = data2[i], g2 = data2[i + 1], b2 = data2[i + 2]
          
          const diff = Math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
          totalDiff += diff
        }
        
        const maxDiff = Math.sqrt(3 * 255 * 255) * (data1.length / 4)
        const similarity = 1 - (totalDiff / maxDiff)
        resolve(Math.max(0, Math.min(1, similarity)))
      }
    }

    img1.crossOrigin = 'anonymous'
    img2.crossOrigin = 'anonymous'
    img1.onload = processImages
    img2.onload = processImages
    img1.onerror = () => resolve(0.5)
    img2.onerror = () => resolve(0.5)
    img1.src = img1Url
    img2.src = img2Url
  })
}

// Calculate structural similarity based on feature distribution
function calculateStructuralSimilarity(features1: number[], features2: number[]): number {
  // Calculate feature statistics
  const mean1 = features1.reduce((sum, val) => sum + val, 0) / features1.length
  const mean2 = features2.reduce((sum, val) => sum + val, 0) / features2.length
  
  const variance1 = features1.reduce((sum, val) => sum + (val - mean1) ** 2, 0) / features1.length
  const variance2 = features2.reduce((sum, val) => sum + (val - mean2) ** 2, 0) / features2.length
  
  // Simplified structural similarity
  const meanSim = 1 - Math.abs(mean1 - mean2)
  const varSim = 1 - Math.abs(variance1 - variance2)
  
  return (meanSim + varSim) / 2
}

// Detect image style based on features
export function detectImageStyle(features: number[]): {
  style: 'cartoon' | 'sketch' | 'realistic' | 'abstract'
  confidence: number
} {
  // Analyze feature distribution
  const mean = features.reduce((sum, val) => sum + val, 0) / features.length
  const variance = features.reduce((sum, val) => sum + (val - mean) ** 2, 0) / features.length
  const sparsity = features.filter(f => Math.abs(f) < 0.01).length / features.length
  const complexity = features.filter(f => Math.abs(f) > 0.1).length / features.length
  
  // Style classification heuristics
  if (sparsity > 0.7 && variance < 0.05) {
    return { style: 'sketch', confidence: 0.8 }
  } else if (complexity < 0.3 && variance > 0.1) {
    return { style: 'cartoon', confidence: 0.75 }
  } else if (complexity > 0.6 && variance > 0.15) {
    return { style: 'realistic', confidence: 0.85 }
  } else {
    return { style: 'abstract', confidence: 0.6 }
  }
}

// Generate enhanced grading with feedback
export function generateEnhancedGrade(
  similarities: {
    cosineSimilarity: number
    ssimScore: number
    structuralSimilarity: number
  },
  style: { style: string; confidence: number },
  config: GradingConfig = defaultGradingConfig
): EnhancedComparisonResult {
  // Combine different similarity metrics
  const combinedScore = (
    similarities.cosineSimilarity * 0.4 +
    similarities.ssimScore * 0.4 +
    similarities.structuralSimilarity * 0.2
  )
  
  // Convert to percentage
  let percentage = Math.max(0, Math.min(100, ((combinedScore + 1) / 2) * 100))
  
  // Apply style-based adjustments
  const styleAdjustment = config.styleAdjustments[style.style as keyof typeof config.styleAdjustments] || 1.0
  percentage *= styleAdjustment
  percentage = Math.min(100, percentage)
  
  // Determine grade
  let grade = 'F'
  let color = 'bg-red-600'
  
  if (percentage >= config.thresholds['A+']) {
    grade = 'A+'
    color = 'bg-green-600'
  } else if (percentage >= config.thresholds.A) {
    grade = 'A'
    color = 'bg-green-500'
  } else if (percentage >= config.thresholds.B) {
    grade = 'B'
    color = 'bg-yellow-500'
  } else if (percentage >= config.thresholds.C) {
    grade = 'C'
    color = 'bg-orange-500'
  } else if (percentage >= config.thresholds.D) {
    grade = 'D'
    color = 'bg-red-400'
  }
  
  // Generate feedback
  const feedback = generateFeedback(similarities, style, percentage)
  
  return {
    similarity: Math.round(percentage),
    ssimScore: Math.round(similarities.ssimScore * 100),
    grade,
    color,
    style: style.style as any,
    styleConfidence: style.confidence,
    feedback
  }
}

// Generate detailed feedback
function generateFeedback(
  similarities: {
    cosineSimilarity: number
    ssimScore: number
    structuralSimilarity: number
  },
  style: { style: string; confidence: number },
  percentage: number
): string[] {
  const feedback: string[] = []
  
  // Overall performance
  if (percentage >= 90) {
    feedback.push("Excellent work! Your drawing closely matches the reference.")
  } else if (percentage >= 80) {
    feedback.push("Great job! Your drawing captures the main elements well.")
  } else if (percentage >= 70) {
    feedback.push("Good effort! There are some similarities with room for improvement.")
  } else if (percentage >= 60) {
    feedback.push("Fair attempt. Focus on matching the key features more closely.")
  } else {
    feedback.push("Keep practicing! Try to observe the reference more carefully.")
  }
  
  // Structural feedback
  if (similarities.structuralSimilarity > 0.7) {
    feedback.push("Good proportions and overall structure.")
  } else if (similarities.structuralSimilarity > 0.5) {
    feedback.push("Structure is recognizable but could be more accurate.")
  } else {
    feedback.push("Focus on getting the basic proportions and layout right.")
  }
  
  // Detail feedback
  if (similarities.ssimScore > 0.8) {
    feedback.push("Excellent attention to detail and texture.")
  } else if (similarities.ssimScore > 0.6) {
    feedback.push("Good detail work with some areas for improvement.")
  } else {
    feedback.push("Try adding more details to match the reference.")
  }
  
  // Style-specific feedback
  if (style.confidence > 0.7) {
    feedback.push(`Well-executed ${style.style} style drawing.`)
  }
  
  return feedback
}

// Generate difference heatmap
export async function generateDifferenceHeatmap(
  img1Url: string,
  img2Url: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      reject(new Error('Could not get canvas context'))
      return
    }
    
    const img1 = new Image()
    const img2 = new Image()
    let loadedCount = 0
    
    const processImages = () => {
      loadedCount++
      if (loadedCount === 2) {
        const size = 256
        canvas.width = canvas.height = size
        
        // Draw and get image data for both images
        ctx.drawImage(img1, 0, 0, size, size)
        const data1 = ctx.getImageData(0, 0, size, size)
        
        ctx.drawImage(img2, 0, 0, size, size)
        const data2 = ctx.getImageData(0, 0, size, size)
        
        // Create difference heatmap
        const diffData = ctx.createImageData(size, size)
        for (let i = 0; i < data1.data.length; i += 4) {
          const diff = Math.abs(data1.data[i] - data2.data[i]) +
                      Math.abs(data1.data[i + 1] - data2.data[i + 1]) +
                      Math.abs(data1.data[i + 2] - data2.data[i + 2])
          
          const intensity = Math.min(255, diff / 3)
          diffData.data[i] = intensity * 2     // Red channel
          diffData.data[i + 1] = 0             // Green channel
          diffData.data[i + 2] = 255 - intensity // Blue channel
          diffData.data[i + 3] = Math.min(255, intensity + 100) // Alpha
        }
        
        ctx.putImageData(diffData, 0, 0)
        resolve(canvas.toDataURL())
      }
    }
    
    img1.crossOrigin = 'anonymous'
    img2.crossOrigin = 'anonymous'
    img1.onload = processImages
    img2.onload = processImages
    img1.onerror = reject
    img2.onerror = reject
    img1.src = img1Url
    img2.src = img2Url
  })
}
