'use client'

import { useState, useRef, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface ImageData {
  file: File
  url: string
  features?: number[]
}

interface ComparisonResult {
  similarity: number
  grade: string
  color: string
}

export default function ImageComparisonApp() {
  const [referenceImage, setReferenceImage] = useState<ImageData | null>(null)
  const [studentImage, setStudentImage] = useState<ImageData | null>(null)
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<ComparisonResult | null>(null)
  const [modelLoading, setModelLoading] = useState(true)

  const referenceInputRef = useRef<HTMLInputElement>(null)
  const studentInputRef = useRef<HTMLInputElement>(null)

  // Load MobileNet model on component mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelLoading(true)
        await tf.ready()
        const loadedModel = await mobilenet.load()
        setModel(loadedModel)
      } catch (error) {
        console.error('Error loading model:', error)
      } finally {
        setModelLoading(false)
      }
    }
    loadModel()
  }, [])

  // Handle image upload
  const handleImageUpload = (file: File, type: 'reference' | 'student') => {
    const url = URL.createObjectURL(file)
    const imageData: ImageData = { file, url }
    
    if (type === 'reference') {
      setReferenceImage(imageData)
    } else {
      setStudentImage(imageData)
    }
    
    // Clear previous results
    setResult(null)
  }

  // Extract features from image using MobileNet
  const extractFeatures = async (imageUrl: string): Promise<number[]> => {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = async () => {
        try {
          if (!model) throw new Error('Model not loaded')
          
          // Convert image to tensor
          const tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .expandDims(0)
            .cast('float32')
            .div(255.0)

          // Get embeddings (features) from the model
          const embeddings = model.infer(tensor, true) as tf.Tensor
          const features = await embeddings.data()
          
          // Clean up tensors
          tensor.dispose()
          embeddings.dispose()
          
          resolve(Array.from(features))
        } catch (error) {
          reject(error)
        }
      }
      img.onerror = reject
      img.src = imageUrl
    })
  }

  // Calculate cosine similarity between two feature vectors
  const cosineSimilarity = (a: number[], b: number[]): number => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0)
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (magnitudeA * magnitudeB)
  }

  // Convert similarity to percentage and assign grade
  const getGradeFromSimilarity = (similarity: number): ComparisonResult => {
    // Convert cosine similarity (-1 to 1) to percentage (0 to 100)
    const percentage = Math.max(0, Math.min(100, ((similarity + 1) / 2) * 100))
    
    let grade: string
    let color: string
    
    if (percentage >= 90) {
      grade = 'A+'
      color = 'bg-green-500'
    } else if (percentage >= 80) {
      grade = 'A'
      color = 'bg-green-400'
    } else if (percentage >= 70) {
      grade = 'B'
      color = 'bg-yellow-500'
    } else if (percentage >= 60) {
      grade = 'C'
      color = 'bg-orange-500'
    } else {
      grade = 'Needs Improvement'
      color = 'bg-red-500'
    }
    
    return { similarity: Math.round(percentage), grade, color }
  }

  // Compare images
  const compareImages = async () => {
    if (!referenceImage || !studentImage || !model) return
    
    setIsLoading(true)
    try {
      // Extract features from both images
      const referenceFeatures = await extractFeatures(referenceImage.url)
      const studentFeatures = await extractFeatures(studentImage.url)
      
      // Calculate similarity
      const similarity = cosineSimilarity(referenceFeatures, studentFeatures)
      const result = getGradeFromSimilarity(similarity)
      
      setResult(result)
    } catch (error) {
      console.error('Error comparing images:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Image Comparison Grader
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered tool to compare reference images with student artwork
          </p>
        </div>

        {modelLoading && (
          <Card className="mb-6">
            <CardContent className="p-6">
              <div className="flex items-center justify-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="text-gray-600">Loading AI model...</span>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <span>Reference Image</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button
                  onClick={() => referenceInputRef.current?.click()}
                  variant="outline"
                  className="w-full h-32 border-2 border-dashed border-gray-300 hover:border-gray-400"
                  disabled={modelLoading}
                >
                  {referenceImage ? 'Change Reference Image' : 'Upload Reference Image'}
                </Button>
                <input
                  ref={referenceInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImageUpload(file, 'reference')
                  }}
                />
                {referenceImage && (
                  <div className="relative">
                    <img
                      src={referenceImage.url}
                      alt="Reference"
                      className="w-full h-48 object-cover rounded-lg border"
                    />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <span>Student Drawing</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button
                  onClick={() => studentInputRef.current?.click()}
                  variant="outline"
                  className="w-full h-32 border-2 border-dashed border-gray-300 hover:border-gray-400"
                  disabled={modelLoading}
                >
                  {studentImage ? 'Change Student Drawing' : 'Upload Student Drawing'}
                </Button>
                <input
                  ref={studentInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImageUpload(file, 'student')
                  }}
                />
                {studentImage && (
                  <div className="relative">
                    <img
                      src={studentImage.url}
                      alt="Student Drawing"
                      className="w-full h-48 object-cover rounded-lg border"
                    />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="text-center mb-8">
          <Button
            onClick={compareImages}
            disabled={!referenceImage || !studentImage || !model || isLoading}
            className="px-8 py-3 text-lg"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Analyzing Images...</span>
              </div>
            ) : (
              'Compare Images'
            )}
          </Button>
        </div>

        {result && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-center">Comparison Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="text-center">
                  <div className="text-6xl font-bold text-gray-900 mb-2">
                    {result.similarity}%
                  </div>
                  <div className="text-lg text-gray-600 mb-4">Similarity Score</div>
                  <Progress value={result.similarity} className="w-full max-w-md mx-auto" />
                </div>

                <div className="text-center">
                  <Badge className={`${result.color} text-white text-xl px-6 py-2`}>
                    {result.grade}
                  </Badge>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-3">Grading Scale:</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-green-500 rounded"></div>
                      <span>A+: 90-100%</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-green-400 rounded"></div>
                      <span>A: 80-89%</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded"></div>
                      <span>B: 70-79%</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-orange-500 rounded"></div>
                      <span>C: 60-69%</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded"></div>
                      <span>Below 60%: Needs Improvement</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {referenceImage && studentImage && (
          <Card>
            <CardHeader>
              <CardTitle className="text-center">Side by Side Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-center mb-3">Reference Image</h3>
                  <img
                    src={referenceImage.url}
                    alt="Reference"
                    className="w-full h-64 object-cover rounded-lg border"
                  />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-center mb-3">Student Drawing</h3>
                  <img
                    src={studentImage.url}
                    alt="Student Drawing"
                    className="w-full h-64 object-cover rounded-lg border"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
