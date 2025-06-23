
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AlertTriangle, CheckCircle, XCircle, Brain, Star } from "lucide-react";
import { toast } from "@/hooks/use-toast";

const ReviewAuthenticator = () => {
  const [reviewText, setReviewText] = useState("");
  const [starRating, setStarRating] = useState(5);
  const [verifiedPurchase, setVerifiedPurchase] = useState("Y");
  const [helpfulVotes, setHelpfulVotes] = useState(0);
  const [totalVotes, setTotalVotes] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);  const analyzeReview = async () => {
    if (!reviewText.trim()) {
      toast({
        title: "Error",
        description: "Please enter a review to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Prepare data for the API
      const requestData = {
        review_text: reviewText,
        star_rating: starRating,
        verified_purchase: verifiedPurchase,
        helpful_votes: helpfulVotes,
        total_votes: totalVotes
      };

      console.log("Sending request to API:", requestData);

      // Send request to Flask API with additional options for CORS
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        mode: 'cors', // Explicitly set CORS mode
        credentials: 'omit', // Don't send credentials
        body: JSON.stringify(requestData),
      });

      console.log("Response status:", response.status);
      console.log("Response headers:", response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Response error:", errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const result = await response.json();
      console.log("API Response:", result);

      // Transform API response to match UI expectations
      const transformedResult = {
        isAuthentic: result.prediction === 1, // 1 = genuine, 0 = fake
        fraudScore: Math.round(result.risk_score * 100), // Convert to percentage
        sentiment: result.confidence.genuine > result.confidence.fake ? "positive" : "negative",
        confidence: Math.round(Math.max(result.confidence.genuine, result.confidence.fake) * 100),
        patterns: result.prediction === 0 ? 
          ["Suspicious patterns detected", "AI model flagged as fake"] : 
          ["Natural language patterns", "AI model verified as genuine"],
        recommendation: result.prediction === 0 ? "Block review" : "Approve review",
        apiResult: result // Store full API response for debugging
      };

      setAnalysisResult(transformedResult);
      
      toast({
        title: "Analysis Complete",
        description: `Review analyzed as ${result.prediction_label.toLowerCase()}`,
        variant: result.prediction === 1 ? "default" : "destructive"
      });

    } catch (error) {
      console.error("Error analyzing review:", error);
      
      // More specific error messages
      let errorMessage = error.message;
      if (error.message.includes('Failed to fetch')) {
        errorMessage = "Cannot connect to API. Please ensure:\n1. Flask API is running on http://127.0.0.1:5000\n2. CORS is properly configured\n3. No firewall blocking the connection";
      }
      
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fillSampleData = () => {
    setReviewText("Using these for years - love them. As a family allergic to wheat, dairy, eggs, nuts, and several other things, we love the entire Cravings Place line of products as it allows us to bake treats with minimal effort and ingredients. Most allergy-free and gluten-free mixes usually just omit one or two allergens at most, so it's great to see a mix created without many of the most common allergens. (Note these still have soy and corn). We consume these on a regular basis and have been doing so for years.");
    setStarRating(5);
    setVerifiedPurchase("Y");
    setHelpfulVotes(0);
    setTotalVotes(0);
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
            </CardTitle>            <CardDescription>
              Analyze reviews using TrustWeaver AI for fraudulent patterns
            </CardDescription>
          </CardHeader>          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div>
                <Label htmlFor="review-text">Review Text</Label>
                <Textarea
                  id="review-text"
                  placeholder="Paste a review to analyze for authenticity..."
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  className="min-h-[120px]"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="star-rating">Star Rating</Label>
                  <Input
                    id="star-rating"
                    type="number"
                    min="1"
                    max="5"
                    value={starRating}
                    onChange={(e) => setStarRating(parseInt(e.target.value) || 1)}
                  />
                </div>
                <div>
                  <Label htmlFor="verified-purchase">Verified Purchase</Label>
                  <select
                    id="verified-purchase"
                    value={verifiedPurchase}
                    onChange={(e) => setVerifiedPurchase(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="Y">Yes</option>
                    <option value="N">No</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="helpful-votes">Helpful Votes</Label>
                  <Input
                    id="helpful-votes"
                    type="number"
                    min="0"
                    value={helpfulVotes}
                    onChange={(e) => setHelpfulVotes(parseInt(e.target.value) || 0)}
                  />
                </div>
                <div>
                  <Label htmlFor="total-votes">Total Votes</Label>
                  <Input
                    id="total-votes"
                    type="number"
                    min="0"
                    value={totalVotes}
                    onChange={(e) => setTotalVotes(parseInt(e.target.value) || 0)}
                  />
                </div>
              </div>

              <div className="flex space-x-2">
                <Button 
                  onClick={analyzeReview} 
                  disabled={isAnalyzing}
                  className="flex-1 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Review"}
                </Button>
                <Button 
                  onClick={fillSampleData}
                  variant="outline"
                  className="border-orange-500 text-orange-500 hover:bg-orange-50"
                >
                  Fill Sample
                </Button>
              </div>
            </div>
              {isAnalyzing && (
              <div className="space-y-2">
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Brain className="h-4 w-4 animate-pulse" />
                  <span>Processing with TrustWeaver AI...</span>
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

                {/* Show extracted features from API */}
                {analysisResult.apiResult?.features_extracted && (
                  <div className="mt-4 p-3 bg-white rounded border">
                    <p className="text-sm text-gray-600 mb-2 font-medium">AI Model Features</p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>Review Length: {analysisResult.apiResult.features_extracted.review_length}</div>
                      <div>Word Count: {analysisResult.apiResult.features_extracted.word_count}</div>
                      <div>Star Rating: {analysisResult.apiResult.features_extracted.star_rating}</div>
                      <div>Verified: {analysisResult.apiResult.features_extracted.is_verified ? 'Yes' : 'No'}</div>
                    </div>
                  </div>
                )}
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
