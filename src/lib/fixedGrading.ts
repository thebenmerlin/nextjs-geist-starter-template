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
    cartoon: 1.05,    // Slightly more lenient for cartoon style
    sketch: 1.03,     // Slightly more lenient for sketches
    realistic: 1.0,   // Standard grading
    abstract: 0.98    // Slightly stricter for abstract
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

// Calculate multiple similarity metrics with MUCH more accurate algorithms
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

  // More accurate cosine similarity
  const cosineSim = calculateAccurateCosineSimilarity(features1, features2)
  
  // Much more accurate SSIM calculation
  const ssimScore = await calculateAccurateSSIM(img1Url, img2Url)
  
  // More discriminating structural similarity
  const structuralSim = calculateAccurateStructuralSimilarity(features1, features2)

  console.log('Raw similarities:', { cosineSim, ssimScore, structuralSim })

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

// Much more accurate cosine similarity calculation
function calculateAccurateCosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0
  
  // Normalize features first to prevent bias
  const normalizeFeatures = (features: number[]) => {
    const mean = features.reduce((sum, val) => sum + val, 0) / features.length
    const std = Math.sqrt(features.reduce((sum, val) => sum + (val - mean) ** 2, 0) / features.length)
    return features.map(val => std > 0 ? (val - mean) / std : 0)
  }
  
  const normA = normalizeFeatures(a)
  const normB = normalizeFeatures(b)
  
  const dotProduct = normA.reduce((sum, val, i) => sum + val * normB[i], 0)
  const magnitudeA = Math.sqrt(normA.reduce((sum, val) => sum + val * val, 0))
  const magnitudeB = Math.sqrt(normB.reduce((sum, val) => sum + val * val, 0))
  
  if (magnitudeA === 0 || magnitudeB === 0) return 0
  
  const similarity = dotProduct / (magnitudeA * magnitudeB)
  
  // Convert from [-1, 1] to [0, 1] and apply more discriminating scaling
  const normalizedSim = (similarity + 1) / 2
  
  // Apply exponential scaling to make differences more pronounced
  return Math.pow(normalizedSim, 2)
}

// Much more accurate SSIM calculation
async function calculateAccurateSSIM(img1Url: string, img2Url: string): Promise<number> {
  return new Promise((resolve) => {
    const canvas1 = document.createElement('canvas')
    const canvas2 = document.createElement('canvas')
    const ctx1 = canvas1.getContext('2d')
    const ctx2 = canvas2.getContext('2d')
    
    if (!ctx1 || !ctx2) {
      resolve(0.1) // Much lower default for failed comparisons
      return
    }

    const img1 = new Image()
    const img2 = new Image()
    let loadedCount = 0

    const processImages = () => {
      loadedCount++
      if (loadedCount === 2) {
        // Use larger size for more accurate comparison
        const size = 128
        canvas1.width = canvas2.width = size
        canvas1.height = canvas2.height = size
        
        ctx1.drawImage(img1, 0, 0, size, size)
        ctx2.drawImage(img2, 0, 0, size, size)
        
        const data1 = ctx1.getImageData(0, 0, size, size).data
        const data2 = ctx2.getImageData(0, 0, size, size).data
        
        // Convert to grayscale and calculate proper SSIM
        const gray1: number[] = []
        const gray2: number[] = []
        
        for (let i = 0; i < data1.length; i += 4) {
          const g1 = 0.299 * data1[i] + 0.587 * data1[i + 1] + 0.114 * data1[i + 2]
          const g2 = 0.299 * data2[i] + 0.587 * data2[i + 1] + 0.114 * data2[i + 2]
          gray1.push(g1)
          gray2.push(g2)
        }
        
        // Calculate means
        const mean1 = gray1.reduce((sum, val) => sum + val, 0) / gray1.length
        const mean2 = gray2.reduce((sum, val) => sum + val, 0) / gray2.length
        
        // Calculate variances
        const var1 = gray1.reduce((sum, val) => sum + (val - mean1) ** 2, 0) / gray1.length
        const var2 = gray2.reduce((sum, val) => sum + (val - mean2) ** 2, 0) / gray2.length
        
        // Calculate covariance
        const covar = gray1.reduce((sum, val, i) => sum + (val - mean1) * (gray2[i] - mean2), 0) / gray1.length
        
        // SSIM constants
        const c1 = (0.01 * 255) ** 2
        const c2 = (0.03 * 255) ** 2
        
        // Calculate SSIM
        const numerator = (2 * mean1 * mean2 + c1) * (2 * covar + c2)
        const denominator = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2)
        
        const ssim = denominator > 0 ? numerator / denominator : 0
        
        // Apply more discriminating scaling
        const clampedSSIM = Math.max(0, Math.min(1, ssim))
        resolve(Math.pow(clampedSSIM, 1.5)) // Make differences more pronounced
      }
    }

    img1.crossOrigin = 'anonymous'
    img2.crossOrigin = 'anonymous'
    img1.onload = processImages
    img2.onload = processImages
    img1.onerror = () => resolve(0.1)
    img2.onerror = () => resolve(0.1)
    img1.src = img1Url
    img2.src = img2Url
  })
}

// Much more discriminating structural similarity
function calculateAccurateStructuralSimilarity(features1: number[], features2: number[]): number {
  if (features1.length !== features2.length) return 0
  
  // Calculate statistical moments
  const mean1 = features1.reduce((sum, val) => sum + val, 0) / features1.length
  const mean2 = features2.reduce((sum, val) => sum + val, 0) / features2.length
  
  const variance1 = features1.reduce((sum, val) => sum + (val - mean1) ** 2, 0) / features1.length
  const variance2 = features2.reduce((sum, val) => sum + (val - mean2) ** 2, 0) / features2.length
  
  const skewness1 = features1.reduce((sum, val) => sum + Math.pow(val - mean1, 3), 0) / (features1.length * Math.pow(variance1, 1.5))
  const skewness2 = features2.reduce((sum, val) => sum + Math.pow(val - mean2, 3), 0) / (features2.length * Math.pow(variance2, 1.5))
  
  // Compare statistical properties
  const meanSim = 1 / (1 + Math.abs(mean1 - mean2) * 10)
  const varSim = 1 / (1 + Math.abs(variance1 - variance2) * 100)
  const skewSim = 1 / (1 + Math.abs(skewness1 - skewness2) * 5)
  
  // Calculate feature correlation
  const correlation = calculateCorrelation(features1, features2)
  
  // Weighted combination with emphasis on correlation
  const structuralSim = (meanSim * 0.2 + varSim * 0.2 + skewSim * 0.2 + correlation * 0.4)
  
  // Apply exponential scaling to make differences more pronounced
  return Math.pow(structuralSim, 2)
}

// Calculate correlation between feature vectors
function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length
  const meanX = x.reduce((sum, val) => sum + val, 0) / n
  const meanY = y.reduce((sum, val) => sum + val, 0) / n
  
  let numerator = 0
  let sumXSquared = 0
  let sumYSquared = 0
  
  for (let i = 0; i < n; i++) {
    const xDiff = x[i] - meanX
    const yDiff = y[i] - meanY
    numerator += xDiff * yDiff
    sumXSquared += xDiff * xDiff
    sumYSquared += yDiff * yDiff
  }
  
  const denominator = Math.sqrt(sumXSquared * sumYSquared)
  if (denominator === 0) return 0
  
  const correlation = numerator / denominator
  return Math.max(0, correlation) // Only positive correlations count as similarity
}

// Detect image style based on features
export function detectImageStyle(features: number[]): {
  style: 'cartoon' | 'sketch' | 'realistic' | 'abstract'
  confidence: number
} {
  // More sophisticated feature analysis
  const mean = features.reduce((sum, val) => sum + val, 0) / features.length
  const variance = features.reduce((sum, val) => sum + (val - mean) ** 2, 0) / features.length
  const sparsity = features.filter(f => Math.abs(f) < 0.001).length / features.length
  const complexity = features.filter(f => Math.abs(f) > 0.05).length / features.length
  const maxActivation = Math.max(...features.map(f => Math.abs(f)))
  
  // More nuanced style classification
  if (sparsity > 0.8 && variance < 0.01 && maxActivation < 0.1) {
    return { style: 'sketch', confidence: 0.85 }
  } else if (complexity < 0.2 && variance > 0.05 && maxActivation < 0.3) {
    return { style: 'cartoon', confidence: 0.8 }
  } else if (complexity > 0.5 && variance > 0.1 && maxActivation > 0.2) {
    return { style: 'realistic', confidence: 0.9 }
  } else {
    return { style: 'abstract', confidence: 0.7 }
  }
}

// Generate enhanced grading with MUCH more accurate scoring
export function generateEnhancedGrade(
  similarities: {
    cosineSimilarity: number
    ssimScore: number
    structuralSimilarity: number
  },
  style: { style: string; confidence: number },
  config: GradingConfig = defaultGradingConfig
): EnhancedComparisonResult {
  console.log('Input similarities:', similarities)
  
  // More sophisticated combination of metrics
  const combinedScore = (
    similarities.cosineSimilarity * 0.5 +    // Deep semantic similarity
    similarities.ssimScore * 0.35 +          // Visual structural similarity  
    similarities.structuralSimilarity * 0.15  // Statistical similarity
  )
  
  console.log('Combined score before scaling:', combinedScore)
  
  // Convert to percentage (already normalized to 0-1)
  let percentage = combinedScore * 100
  
  // Apply style-based adjustments (much smaller adjustments)
  const styleAdjustment = config.styleAdjustments[style.style as keyof typeof config.styleAdjustments] || 1.0
  percentage *= styleAdjustment
  
  // Ensure percentage is within bounds
  percentage = Math.max(0, Math.min(100, percentage))
  
  console.log('Final percentage:', percentage)
  
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
  
  // Generate more accurate feedback
  const feedback = generateAccurateFeedback(similarities, style, percentage)
  
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

// Generate more accurate and helpful feedback
function generateAccurateFeedback(
  similarities: {
    cosineSimilarity: number
    ssimScore: number
    structuralSimilarity: number
  },
  style: { style: string; confidence: number },
  percentage: number
): string[] {
  const feedback: string[] = []
  
  // Overall performance with more realistic thresholds
  if (percentage >= 85) {
    feedback.push("Outstanding work! Your drawing is remarkably similar to the reference.")
  } else if (percentage >= 70) {
    feedback.push("Great job! Your drawing captures most key elements effectively.")
  } else if (percentage >= 55) {
    feedback.push("Good effort! You've captured some important similarities.")
  } else if (percentage >= 40) {
    feedback.push("Fair attempt. Focus on matching the main shapes and proportions.")
  } else if (percentage >= 25) {
    feedback.push("Keep practicing! Try to observe the reference more carefully.")
  } else {
    feedback.push("This appears quite different from the reference. Study the original more closely.")
  }
  
  // Specific metric feedback
  if (similarities.cosineSimilarity < 0.3) {
    feedback.push("The overall content and subject matter differ significantly from the reference.")
  } else if (similarities.cosineSimilarity < 0.6) {
    feedback.push("Some elements match the reference, but key features are missing or different.")
  }
  
  if (similarities.ssimScore < 0.4) {
    feedback.push("The visual structure and layout need significant improvement.")
  } else if (similarities.ssimScore < 0.7) {
    feedback.push("Good structural foundation, but details could be more accurate.")
  }
  
  if (similarities.structuralSimilarity < 0.4) {
    feedback.push("Focus on matching the basic proportions and spatial relationships.")
  }
  
  // Style-specific feedback
  if (style.confidence > 0.7) {
    feedback.push(`Nice ${style.style} style execution.`)
  }
  
  return feedback
}

// Generate difference heatmap (unchanged but more accurate)
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
        
        // Create more accurate difference heatmap
        const diffData = ctx.createImageData(size, size)
        for (let i = 0; i < data1.data.length; i += 4) {
          const diff = Math.abs(data1.data[i] - data2.data[i]) +
                      Math.abs(data1.data[i + 1] - data2.data[i + 1]) +
                      Math.abs(data1.data[i + 2] - data2.data[i + 2])
          
          const intensity = Math.min(255, diff / 3)
          const normalizedIntensity = intensity / 255
          
          // More pronounced color mapping
          diffData.data[i] = Math.min(255, intensity * 1.5)     // Red channel
          diffData.data[i + 1] = Math.max(0, 128 - intensity)   // Green channel
          diffData.data[i + 2] = Math.max(0, 255 - intensity * 2) // Blue channel
          diffData.data[i + 3] = Math.min(255, intensity + 50)  // Alpha
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
