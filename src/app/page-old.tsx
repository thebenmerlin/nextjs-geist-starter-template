'use client'

import { useState, useRef, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  calculateEnhancedSimilarity, 
  detectImageStyle, 
  generateEnhancedGrade, 
  generateDifferenceHeatmap,
  EnhancedComparisonResult,
  defaultGradingConfig
} from '@/lib/advancedGrading'

interface ImageData {
  file: File
  url: string
}

export default function EnhancedImageComparisonApp() {
  const [referenceImage, setReferenceImage] = useState<ImageData | null>(null)
  const [studentImage, setStudentImage] = useState<ImageData | null>(null)
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<EnhancedComparisonResult | null>(null)
  const [modelLoading, setModelLoading] = useState(true)
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null)

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
    setHeatmapUrl(null)
  }

  // Enhanced image comparison
  const compareImages = async () => {
    if (!referenceImage || !studentImage || !model) return
    
    setIsLoading(true)
    try {
      // Calculate enhanced similarity metrics
      const similarities = await calculateEnhancedSimilarity(
        referenceImage.url,
        studentImage.url,
        model
      )

      // Extract features for style detection
      const features = await extractFeatures(referenceImage.url, model)
      const style = detectImageStyle(features)

      // Generate enhanced grade with feedback
      const enhancedResult = generateEnhancedGrade(similarities, style, defaultGradingConfig)

      // Generate difference heatmap
      const heatmap = await generateDifferenceHeatmap(referenceImage.url, studentImage.url)
      
      setResult(enhancedResult)
      setHeatmapUrl(heatmap)
    } catch (error) {
      console.error('Error comparing images:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // Extract features helper function
  const extractFeatures = async (imageUrl: string, model: any): Promise<number[]> => {
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Advanced Image Comparison Grader
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered tool with enhanced grading, style detection, and detailed feedback
          </p>
        </div>

        {/* Model Loading Status */}
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

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Reference Image Upload */}
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

          {/* Student Image Upload */}
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

        {/* Compare Button */}
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

        {/* Enhanced Results Section */}
        {result && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-center">Enhanced Comparison Results</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="feedback">Feedback</TabsTrigger>
                  <TabsTrigger value="heatmap">Analysis</TabsTrigger>
                </TabsList>

                {/* Overview Tab */}
                <TabsContent value="overview" className="space-y-6">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-gray-900 mb-2">
                      {result.similarity}%
                    </div>
                    <div className="text-lg text-gray-600 mb-4">Overall Similarity Score</div>
                    <Progress value={result.similarity} className="w-full max-w-md mx-auto mb-4" />
                    
                    <div className="flex justify-center items-center space-x-4 mb-4">
                      <Badge className={`${result.color} text-white text-xl px-6 py-2`}>
                        {result.grade}
                      </Badge>
                      <Badge variant="outline" className="text-lg px-4 py-1">
                        {result.style} style ({Math.round(result.styleConfidence * 100)}% confidence)
                      </Badge>
                    </div>
                  </div>
                </TabsContent>

                {/* Metrics Tab */}
                <TabsContent value="metrics" className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Structural Similarity (SSIM)</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-blue-600 mb-2">
                          {result.ssimScore}%
                        </div>
                        <Progress value={result.ssimScore} className="mb-2" />
                        <p className="text-sm text-gray-600">
                          Measures structural and textural similarity
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Style Detection</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold text-purple-600 mb-2 capitalize">
                          {result.style}
                        </div>
                        <Progress value={result.styleConfidence * 100} className="mb-2" />
                        <p className="text-sm text-gray-600">
                          Detected drawing style with confidence level
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Enhanced Grading Scale */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-gray-900 mb-3">Enhanced Grading Scale:</h3>
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-sm">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-600 rounded"></div>
                        <span>A+: 90-100%</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded"></div>
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
                        <div className="w-3 h-3 bg-red-400 rounded"></div>
                        <span>D: 50-59%</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-red-600 rounded"></div>
                        <span>F: Below 50%</span>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Feedback Tab */}
                <TabsContent value="feedback" className="space-y-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-blue-900 mb-3">Detailed Feedback:</h3>
                    <ul className="space-y-2">
                      {result.feedback.map((item, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-blue-600 mt-1">•</span>
                          <span className="text-blue-800">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </TabsContent>

                {/* Heatmap Tab */}
                <TabsContent value="heatmap" className="space-y-4">
                  {heatmapUrl && (
                    <div className="text-center">
                      <h3 className="font-semibold text-gray-900 mb-4">Difference Heatmap</h3>
                      <p className="text-sm text-gray-600 mb-4">
                        Red areas show the biggest differences between images
                      </p>
                      <img
                        src={heatmapUrl}
                        alt="Difference Heatmap"
                        className="max-w-md mx-auto rounded-lg border shadow-lg"
                      />
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

        {/* Side by Side Comparison */}
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
