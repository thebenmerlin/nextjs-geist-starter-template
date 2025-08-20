import * as tf from '@tensorflow/tfjs'

// SSIM (Structural Similarity Index) calculation
export function calculateSSIM(img1: tf.Tensor, img2: tf.Tensor): number {
  // Ensure images are the same size
  const [height, width] = img1.shape.slice(0, 2)
  const resizedImg2 = tf.image.resizeBilinear(img2 as tf.Tensor3D, [height, width])
  
  // Convert to grayscale if needed
  const gray1 = img1.shape[2] === 3 ? tf.image.rgbToGrayscale(img1 as tf.Tensor3D) : img1
  const gray2 = resizedImg2.shape[2] === 3 ? tf.image.rgbToGrayscale(resizedImg2 as tf.Tensor3D) : resizedImg2
  
  // Calculate means
  const mu1 = tf.mean(gray1)
  const mu2 = tf.mean(gray2)
  
  // Calculate variances and covariance
  const mu1Sq = tf.square(mu1)
  const mu2Sq = tf.square(mu2)
  const mu1Mu2 = tf.mul(mu1, mu2)
  
  const sigma1Sq = tf.mean(tf.square(gray1)).sub(mu1Sq)
  const sigma2Sq = tf.mean(tf.square(gray2)).sub(mu2Sq)
  const sigma12 = tf.mean(tf.mul(gray1, gray2)).sub(mu1Mu2)
  
  // SSIM constants
  const c1 = 0.01 * 0.01
  const c2 = 0.03 * 0.03
  
  // Calculate SSIM
  const numerator = tf.mul(tf.add(tf.mul(mu1Mu2, 2), c1), tf.add(tf.mul(sigma12, 2), c2))
  const denominator = tf.mul(tf.add(mu1Sq.add(mu2Sq), c1), tf.add(sigma1Sq.add(sigma2Sq), c2))
  
  const ssim = tf.div(numerator, denominator)
  const result = ssim.dataSync()[0]
  
  // Clean up tensors
  gray1.dispose()
  gray2.dispose()
  resizedImg2.dispose()
  mu1.dispose()
  mu2.dispose()
  mu1Sq.dispose()
  mu2Sq.dispose()
  mu1Mu2.dispose()
  sigma1Sq.dispose()
  sigma2Sq.dispose()
  sigma12.dispose()
  numerator.dispose()
  denominator.dispose()
  ssim.dispose()
  
  return result
}

// Enhanced feature extraction with multiple layers
export async function extractEnhancedFeatures(imageUrl: string, model: any): Promise<{
  deepFeatures: number[]
  colorHistogram: number[]
  edgeFeatures: number[]
}> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = async () => {
      try {
        // Convert image to tensor
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .expandDims(0)
          .cast('float32')
          .div(255.0)

        // Deep features from MobileNet
        const embeddings = model.infer(tensor, true) as tf.Tensor
        const deepFeatures = Array.from(await embeddings.data())

        // Color histogram features
        const colorHistogram = await extractColorHistogram(tensor)
        
        // Edge detection features
        const edgeFeatures = await extractEdgeFeatures(tensor)

        // Clean up
        tensor.dispose()
        embeddings.dispose()

        resolve({
          deepFeatures,
          colorHistogram,
          edgeFeatures
        })
      } catch (error) {
        reject(error)
      }
    }
    img.onerror = reject
    img.src = imageUrl
  })
}

// Extract color histogram
async function extractColorHistogram(tensor: tf.Tensor): Promise<number[]> {
  const [r, g, b] = tf.split(tensor.squeeze(), 3, 3)
  
  // Calculate histograms for each channel (simplified)
  const rHist = tf.mean(r, [0, 1])
  const gHist = tf.mean(g, [0, 1])
  const bHist = tf.mean(b, [0, 1])
  
  const histogram = tf.concat([rHist, gHist, bHist])
  const result = Array.from(await histogram.data())
  
  // Clean up
  r.dispose()
  g.dispose()
  b.dispose()
  rHist.dispose()
  gHist.dispose()
  bHist.dispose()
  histogram.dispose()
  
  return result
}

// Extract edge features using Sobel filter
async function extractEdgeFeatures(tensor: tf.Tensor): Promise<number[]> {
  const grayscale = tf.image.rgbToGrayscale(tensor.squeeze())
  
  // Sobel X kernel
  const sobelX = tf.tensor2d([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ]).expandDims(2).expandDims(3)
  
  // Sobel Y kernel
  const sobelY = tf.tensor2d([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
  ]).expandDims(2).expandDims(3)
  
  // Apply convolution
  const edgesX = tf.conv2d(grayscale.expandDims(0).expandDims(3), sobelX, 1, 'same')
  const edgesY = tf.conv2d(grayscale.expandDims(0).expandDims(3), sobelY, 1, 'same')
  
  // Calculate edge magnitude
  const edgeMagnitude = tf.sqrt(tf.add(tf.square(edgesX), tf.square(edgesY)))
  const edgeFeatures = tf.mean(edgeMagnitude, [0, 1, 2])
  
  const result = Array.from(await edgeFeatures.data())
  
  // Clean up
  grayscale.dispose()
  sobelX.dispose()
  sobelY.dispose()
  edgesX.dispose()
  edgesY.dispose()
  edgeMagnitude.dispose()
  edgeFeatures.dispose()
  
  return result
}

// Style detection (simplified classification)
export function detectImageStyle(features: {
  deepFeatures: number[]
  colorHistogram: number[]
  edgeFeatures: number[]
}): {
  style: 'cartoon' | 'sketch' | 'realistic' | 'abstract'
  confidence: number
} {
  const { deepFeatures, colorHistogram, edgeFeatures } = features
  
  // Simplified heuristics for style detection
  const colorVariance = colorHistogram.reduce((sum, val, i, arr) => {
    const mean = arr.reduce((a, b) => a + b) / arr.length
    return sum + Math.pow(val - mean, 2)
  }, 0) / colorHistogram.length
  
  const edgeIntensity = edgeFeatures.reduce((sum, val) => sum + val, 0) / edgeFeatures.length
  const featureComplexity = deepFeatures.filter(f => Math.abs(f) > 0.1).length / deepFeatures.length
  
  // Style classification logic
  if (edgeIntensity > 0.3 && colorVariance < 0.1) {
    return { style: 'sketch', confidence: 0.8 }
  } else if (colorVariance > 0.2 && featureComplexity < 0.3) {
    return { style: 'cartoon', confidence: 0.75 }
  } else if (featureComplexity > 0.6 && colorVariance > 0.15) {
    return { style: 'realistic', confidence: 0.85 }
  } else {
    return { style: 'abstract', confidence: 0.6 }
  }
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
        // Set canvas size
        canvas.width = Math.max(img1.width, img2.width)
        canvas.height = Math.max(img1.height, img2.height)
        
        // Draw first image
        ctx.drawImage(img1, 0, 0, canvas.width, canvas.height)
        const imageData1 = ctx.getImageData(0, 0, canvas.width, canvas.height)
        
        // Draw second image
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img2, 0, 0, canvas.width, canvas.height)
        const imageData2 = ctx.getImageData(0, 0, canvas.width, canvas.height)
        
        // Calculate difference
        const diffData = ctx.createImageData(canvas.width, canvas.height)
        for (let i = 0; i < imageData1.data.length; i += 4) {
          const diff = Math.abs(imageData1.data[i] - imageData2.data[i]) +
                      Math.abs(imageData1.data[i + 1] - imageData2.data[i + 1]) +
                      Math.abs(imageData1.data[i + 2] - imageData2.data[i + 2])
          
          const intensity = Math.min(255, diff)
          diffData.data[i] = intensity     // Red
          diffData.data[i + 1] = 0         // Green
          diffData.data[i + 2] = 255 - intensity // Blue
          diffData.data[i + 3] = Math.min(255, intensity * 2) // Alpha
        }
        
        // Draw difference heatmap
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
