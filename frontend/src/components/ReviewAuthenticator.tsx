
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle, XCircle, Brain, Star } from "lucide-react";
import { toast } from "@/hooks/use-toast";

const ReviewAuthenticator = () => {
  const [reviewText, setReviewText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const analyzeReview = async () => {
    if (!reviewText.trim()) {
      toast({
        title: "Error",
        description: "Please enter a review to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate AI analysis
    setTimeout(() => {
      const suspiciousPatterns = [
        "fake", "bot", "paid", "incentive", "discount", "free product"
      ];
      
      const isSuspicious = suspiciousPatterns.some(pattern => 
        reviewText.toLowerCase().includes(pattern)
      );
      
      const sentiment = reviewText.length > 100 ? 
        (Math.random() > 0.5 ? "positive" : "negative") : "neutral";
      
      const fraudScore = isSuspicious ? 
        Math.floor(Math.random() * 40) + 60 : 
        Math.floor(Math.random() * 30) + 10;

      setAnalysisResult({
        isAuthentic: !isSuspicious,
        fraudScore,
        sentiment,
        confidence: Math.floor(Math.random() * 20) + 80,
        patterns: isSuspicious ? ["Incentivized review patterns", "Unnatural language"] : ["Natural language flow", "Balanced sentiment"],
        recommendation: isSuspicious ? "Block review" : "Approve review"
      });
      
      setIsAnalyzing(false);
      
      toast({
        title: "Analysis Complete",
        description: `Review analyzed with ${!isSuspicious ? 'authentic' : 'suspicious'} result`,
        variant: !isSuspicious ? "default" : "destructive"
      });
    }, 2000);
  };

  const mockReviews = [
    {
      id: 1,
      text: "This product is amazing! I received it for free in exchange for an honest review...",
      rating: 5,
      status: "flagged",
      fraudScore: 87
    },
    {
      id: 2,
      text: "Good quality product, fast shipping. The item matches the description perfectly.",
      rating: 4,
      status: "approved",
      fraudScore: 15
    },
    {
      id: 3,
      text: "Best product ever!!! 5 stars all the way! You should definitely buy this now!!!",
      rating: 5,
      status: "flagged",
      fraudScore: 92
    }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Review Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-orange-500" />
              <span>Review Authentication</span>
            </CardTitle>
            <CardDescription>
              Analyze reviews using Bedrock LLMs for fraudulent patterns
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Paste a review to analyze for authenticity..."
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              className="min-h-[120px]"
            />
            <Button 
              onClick={analyzeReview} 
              disabled={isAnalyzing}
              className="w-full bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Review"}
            </Button>
            
            {isAnalyzing && (
              <div className="space-y-2">
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Brain className="h-4 w-4 animate-pulse" />
                  <span>Processing with Bedrock LLM...</span>
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
                      <XCircle className="h-5 w-5 text-red-500" />
                    )}
                    <span className="font-semibold">
                      {analysisResult.isAuthentic ? "Authentic Review" : "Suspicious Review"}
                    </span>
                  </div>
                  <Badge variant={analysisResult.isAuthentic ? "default" : "destructive"}>
                    {analysisResult.recommendation}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Fraud Score</p>
                    <div className="flex items-center space-x-2">
                      <Progress value={analysisResult.fraudScore} className="flex-1 h-2" />
                      <span className="text-sm font-medium">{analysisResult.fraudScore}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Confidence</p>
                    <p className="text-lg font-semibold">{analysisResult.confidence}%</p>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600 mb-2">Detected Patterns</p>
                  <div className="space-y-1">
                    {analysisResult.patterns.map((pattern, index) => (
                      <div key={index} className="flex items-center space-x-2 text-sm">
                        <div className={`w-2 h-2 rounded-full ${analysisResult.isAuthentic ? 'bg-green-500' : 'bg-red-500'}`} />
                        <span>{pattern}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Reviews */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Review Analysis</CardTitle>
            <CardDescription>Live monitoring of marketplace reviews</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {mockReviews.map((review) => (
                <div key={review.id} className="p-3 border rounded-lg">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className="flex">
                        {[...Array(5)].map((_, i) => (
                          <Star
                            key={i}
                            className={`h-4 w-4 ${
                              i < review.rating ? 'text-yellow-400 fill-current' : 'text-gray-300'
                            }`}
                          />
                        ))}
                      </div>
                      <Badge 
                        variant={review.status === 'approved' ? 'default' : 'destructive'}
                        className="text-xs"
                      >
                        {review.status}
                      </Badge>
                    </div>
                    <span className="text-xs text-gray-500">Score: {review.fraudScore}%</span>
                  </div>
                  <p className="text-sm text-gray-700 line-clamp-2">{review.text}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ReviewAuthenticator;
