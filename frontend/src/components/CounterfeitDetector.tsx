
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, Eye, AlertTriangle, CheckCircle, Image } from "lucide-react";
import { toast } from "@/hooks/use-toast";

const CounterfeitDetector = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = () => {
    if (!uploadedImage) {
      toast({
        title: "Error",
        description: "Please upload an image to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate AI image analysis
    setTimeout(() => {
      const isCounterfeit = Math.random() > 0.7;
      const confidence = Math.floor(Math.random() * 20) + 80;
      
      setAnalysisResult({
        isAuthentic: !isCounterfeit,
        confidence,
        matchScore: isCounterfeit ? Math.floor(Math.random() * 30) + 20 : Math.floor(Math.random() * 20) + 80,
        detectedFeatures: [
          "Logo authenticity",
          "Packaging quality",
          "Text alignment",
          "Color accuracy",
          "Material texture"
        ],
        issues: isCounterfeit ? ["Logo inconsistencies", "Poor print quality"] : [],
        recommendation: isCounterfeit ? "Block listing" : "Approve listing"
      });
      
      setIsAnalyzing(false);
      
      toast({
        title: "Analysis Complete",
        description: `Product ${!isCounterfeit ? 'authentic' : 'potentially counterfeit'}`,
        variant: !isCounterfeit ? "default" : "destructive"
      });
    }, 3000);
  };

  const flaggedProducts = [
    {
      id: 1,
      name: "Designer Handbag",
      image: "/api/placeholder/80/80",
      issues: ["Logo mismatch", "Poor stitching"],
      confidence: 94,
      status: "blocked"
    },
    {
      id: 2,
      name: "Electronics Component",
      image: "/api/placeholder/80/80",
      issues: ["Packaging inconsistency"],
      confidence: 87,
      status: "review"
    },
    {
      id: 3,
      name: "Luxury Watch",
      image: "/api/placeholder/80/80",
      issues: ["Material analysis failed"],
      confidence: 91,
      status: "blocked"
    }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Eye className="h-5 w-5 text-blue-500" />
              <span>Counterfeit Detection</span>
            </CardTitle>
            <CardDescription>
              Image analysis using Amazon Rekognition Custom Labels
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              {uploadedImage ? (
                <div className="space-y-4">
                  <img 
                    src={uploadedImage} 
                    alt="Uploaded product" 
                    className="max-h-48 mx-auto rounded-lg"
                  />
                  <Button 
                    variant="outline" 
                    onClick={() => document.getElementById('image-upload').click()}
                  >
                    Change Image
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-sm text-gray-600 mb-2">
                      Upload product image for authenticity verification
                    </p>
                    <Button 
                      onClick={() => document.getElementById('image-upload').click()}
                      className="bg-blue-500 hover:bg-blue-600"
                    >
                      Upload Image
                    </Button>
                  </div>
                </div>
              )}
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>

            <Button 
              onClick={analyzeImage} 
              disabled={isAnalyzing || !uploadedImage}
              className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Product"}
            </Button>

            {isAnalyzing && (
              <div className="space-y-2">
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Eye className="h-4 w-4 animate-pulse" />
                  <span>Processing with Amazon Rekognition...</span>
                </div>
                <Progress value={Math.random() * 100} className="h-2" />
              </div>
            )}

            {analysisResult && (
              <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {analysisResult.isAuthentic ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    )}
                    <span className="font-semibold">
                      {analysisResult.isAuthentic ? "Authentic Product" : "Potential Counterfeit"}
                    </span>
                  </div>
                  <Badge variant={analysisResult.isAuthentic ? "default" : "destructive"}>
                    {analysisResult.recommendation}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Match Score</p>
                    <div className="flex items-center space-x-2">
                      <Progress value={analysisResult.matchScore} className="flex-1 h-2" />
                      <span className="text-sm font-medium">{analysisResult.matchScore}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Confidence</p>
                    <p className="text-lg font-semibold">{analysisResult.confidence}%</p>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600 mb-2">Analyzed Features</p>
                  <div className="grid grid-cols-2 gap-2">
                    {analysisResult.detectedFeatures.map((feature, index) => (
                      <div key={index} className="flex items-center space-x-2 text-sm">
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {analysisResult.issues.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-600 mb-2">Detected Issues</p>
                    <div className="space-y-1">
                      {analysisResult.issues.map((issue, index) => (
                        <div key={index} className="flex items-center space-x-2 text-sm text-red-600">
                          <AlertTriangle className="h-3 w-3" />
                          <span>{issue}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Flagged Products */}
        <Card>
          <CardHeader>
            <CardTitle>Flagged Products</CardTitle>
            <CardDescription>Recently detected counterfeit items</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {flaggedProducts.map((product) => (
                <div key={product.id} className="p-3 border rounded-lg">
                  <div className="flex items-start space-x-3">
                    <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center">
                      <Image className="h-8 w-8 text-gray-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-sm truncate">{product.name}</h4>
                        <Badge 
                          variant={product.status === 'blocked' ? 'destructive' : 'secondary'}
                          className="text-xs"
                        >
                          {product.status}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-500 mb-2">
                        Confidence: {product.confidence}%
                      </p>
                      <div className="space-y-1">
                        {product.issues.map((issue, index) => (
                          <div key={index} className="flex items-center space-x-2 text-xs text-red-600">
                            <div className="w-1 h-1 bg-red-500 rounded-full" />
                            <span>{issue}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default CounterfeitDetector;
