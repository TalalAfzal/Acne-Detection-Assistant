import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    pipeline
)
from typing import List, Dict, Optional, Any
import random
import re
from knowledge_base import KnowledgeBase

class AcneChatbot:
    """AI-powered chatbot for acne treatment advice using RAG"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", knowledge_base_file: str = "acne_knowledge_base.json"):
        """
        Initialize the acne chatbot
        
        Args:
            model_name: Hugging Face model name for conversation
            knowledge_base_file: Path to knowledge base file
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(knowledge_base_file)
        
        # Initialize conversation model
        self.tokenizer = None
        self.model = None
        self.conversation_pipeline = None
        
        # Initialize models
        self._initialize_models()
        
        # Conversation context
        self.conversation_history = []
        self.user_context = {
            'detected_acne': None,
            'severity': None,
            'previous_treatments': [],
            'skin_type': None
        }
        
        # Response templates
        self.greeting_responses = [
            "Hello! I'm here to help you with acne treatment advice. How can I assist you today?",
            "Hi there! I'm your acne treatment assistant. What would you like to know about acne care?",
            "Welcome! I'm here to provide personalized acne treatment recommendations. How can I help?"
        ]
        
        self.fallback_responses = [
            "I understand you're asking about acne treatment. Let me search for relevant information to help you.",
            "That's a great question about acne care. Let me find the best advice for your situation.",
            "I want to make sure I give you accurate information. Let me look up the most current treatment recommendations."
        ]
    
    def _initialize_models(self):
        """
        Initialize the conversation models
        """
        try:
            # Try to initialize a lightweight conversational model
            self.conversation_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer="microsoft/DialoGPT-small",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Conversation model initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize conversation model: {e}")
            print("Falling back to template-based responses")
            self.conversation_pipeline = None
    
    def get_response(self, user_input: str, detected_acne: Optional[List[Dict]] = None) -> str:
        """
        Generate response to user input using RAG approach
        
        Args:
            user_input: User's message
            detected_acne: Optional acne detection results
            
        Returns:
            Chatbot response
        """
        # Update user context
        if detected_acne:
            self.user_context['detected_acne'] = detected_acne
            self.user_context['severity'] = self._determine_overall_severity(detected_acne)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Analyze user intent
            intent = self._analyze_intent(user_input)
            
            # Retrieve relevant information from knowledge base
            relevant_info = self._retrieve_relevant_info(user_input, intent)
            
            # Generate response
            response = self._generate_response(user_input, intent, relevant_info)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._get_fallback_response()
    
    def _analyze_intent(self, user_input: str) -> str:
        """
        Analyze user intent from input
        
        Args:
            user_input: User's message
            
        Returns:
            Detected intent
        """
        user_input_lower = user_input.lower()
        
        # Define intent patterns
        intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'treatment_inquiry': ['treatment', 'cure', 'medicine', 'medication', 'how to treat'],
            'product_recommendation': ['recommend', 'suggest', 'best product', 'what should i use'],
            'routine_inquiry': ['routine', 'skincare', 'daily care', 'how often'],
            'severity_question': ['severe', 'mild', 'moderate', 'how bad', 'serious'],
            'cause_inquiry': ['why', 'cause', 'reason', 'what causes'],
            'prevention': ['prevent', 'avoid', 'stop', 'prevention'],
            'side_effects': ['side effect', 'reaction', 'irritation', 'allergy'],
            'dermatologist': ['dermatologist', 'doctor', 'professional', 'specialist'],
            'lifestyle': ['diet', 'food', 'exercise', 'stress', 'lifestyle']
        }
        
        # Check for patterns
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    return intent
        
        return 'general_inquiry'
    
    def _retrieve_relevant_info(self, user_input: str, intent: str) -> Dict[str, Any]:
        """
        Retrieve relevant information from knowledge base
        
        Args:
            user_input: User's message
            intent: Detected intent
            
        Returns:
            Dictionary of relevant information
        """
        relevant_info = {
            'treatments': [],
            'routine': {},
            'tips': [],
            'referral_criteria': [],
            'myths_facts': []
        }
        
        try:
            # Search for relevant treatments
            relevant_info['treatments'] = self.knowledge_base.search_treatments(user_input, method="hybrid", top_k=3)
            
            # Get severity-specific treatments if we have detection results
            if self.user_context['severity']:
                severity_treatments = self.knowledge_base.get_treatment_by_severity(self.user_context['severity'])
                relevant_info['treatments'].extend(severity_treatments[:2])
            
            # Get routine information for routine inquiries
            if intent == 'routine_inquiry':
                relevant_info['routine'] = self.knowledge_base.get_skincare_routine()
            
            # Get lifestyle tips for lifestyle inquiries
            if intent == 'lifestyle':
                relevant_info['tips'] = self.knowledge_base.get_lifestyle_tips()
            
            # Get dermatologist referral criteria
            if intent == 'dermatologist':
                relevant_info['referral_criteria'] = self.knowledge_base.get_dermatologist_referral_criteria()
            
            # Get myths and facts for cause inquiries
            if intent == 'cause_inquiry':
                relevant_info['myths_facts'] = self.knowledge_base.get_myths_and_facts()
        
        except Exception as e:
            print(f"Error retrieving information: {e}")
        
        return relevant_info
    
    def _generate_response(self, user_input: str, intent: str, relevant_info: Dict[str, Any]) -> str:
        """
        Generate response based on intent and retrieved information
        
        Args:
            user_input: User's message
            intent: Detected intent
            relevant_info: Retrieved information from knowledge base
            
        Returns:
            Generated response
        """
        if intent == 'greeting':
            return random.choice(self.greeting_responses)
        
        elif intent == 'treatment_inquiry':
            return self._generate_treatment_response(relevant_info['treatments'])
        
        elif intent == 'product_recommendation':
            return self._generate_product_recommendation(relevant_info['treatments'])
        
        elif intent == 'routine_inquiry':
            return self._generate_routine_response(relevant_info['routine'])
        
        elif intent == 'severity_question':
            return self._generate_severity_response()
        
        elif intent == 'cause_inquiry':
            return self._generate_cause_response(relevant_info['myths_facts'])
        
        elif intent == 'prevention':
            return self._generate_prevention_response(relevant_info['tips'])
        
        elif intent == 'dermatologist':
            return self._generate_dermatologist_response(relevant_info['referral_criteria'])
        
        elif intent == 'lifestyle':
            return self._generate_lifestyle_response(relevant_info['tips'])
        
        else:
            return self._generate_general_response(user_input, relevant_info)
    
    def _generate_treatment_response(self, treatments: List[Dict]) -> str:
        """
        Generate treatment recommendation response
        
        Args:
            treatments: List of relevant treatments
            
        Returns:
            Treatment response
        """
        if not treatments:
            return "I'd be happy to help with treatment recommendations. Could you tell me more about your acne severity or specific concerns?"
        
        response = "Based on your inquiry, here are some treatment options I'd recommend:\n\n"
        
        for i, treatment in enumerate(treatments[:3], 1):
            response += f"{i}. **{treatment['name']}** ({treatment['type']} treatment)\n"
            response += f"   - {treatment['description']}\n"
            response += f"   - Usage: {treatment['usage']}\n"
            
            if treatment.get('benefits'):
                response += f"   - Benefits: {', '.join(treatment['benefits'][:3])}\n"
            
            response += "\n"
        
        response += "💡 **Important**: Always consult with a dermatologist before starting new treatments, especially for moderate to severe acne."
        
        return response
    
    def _generate_product_recommendation(self, treatments: List[Dict]) -> str:
        """
        Generate product recommendation response
        
        Args:
            treatments: List of relevant treatments
            
        Returns:
            Product recommendation response
        """
        if not treatments:
            return "I'd love to help you find the right products! Could you tell me about your skin type and current acne severity?"
        
        # Focus on topical treatments for product recommendations
        topical_treatments = [t for t in treatments if t.get('type') == 'topical']
        
        if not topical_treatments:
            topical_treatments = treatments[:2]
        
        response = "Here are my top product recommendations for you:\n\n"
        
        for treatment in topical_treatments[:2]:
            response += f"🔹 **{treatment['name']}**\n"
            response += f"   Perfect for: {', '.join(treatment.get('severity', ['general use']))} acne\n"
            response += f"   How to use: {treatment['usage']}\n"
            
            if treatment.get('precautions'):
                response += f"   ⚠️ Note: {treatment['precautions']}\n"
            
            response += "\n"
        
        response += "Remember to introduce new products gradually and patch test first!"
        
        return response
    
    def _generate_routine_response(self, routine: Dict) -> str:
        """
        Generate skincare routine response
        
        Args:
            routine: Skincare routine information
            
        Returns:
            Routine response
        """
        if not routine:
            return "A good skincare routine is essential for managing acne. Let me get the routine recommendations for you."
        
        response = "Here's a comprehensive skincare routine for acne-prone skin:\n\n"
        
        if 'morning' in routine:
            response += "🌅 **Morning Routine:**\n"
            for step in routine['morning']:
                response += f"{step['step']}. {step['product']}: {step['description']}\n"
                if step.get('tips'):
                    response += f"   💡 Tip: {step['tips']}\n"
            response += "\n"
        
        if 'evening' in routine:
            response += "🌙 **Evening Routine:**\n"
            for step in routine['evening']:
                response += f"{step['step']}. {step['product']}: {step['description']}\n"
                if step.get('tips'):
                    response += f"   💡 Tip: {step['tips']}\n"
            response += "\n"
        
        response += "Consistency is key! Stick to your routine for at least 6-8 weeks to see results."
        
        return response
    
    def _generate_severity_response(self) -> str:
        """
        Generate response about acne severity
        
        Returns:
            Severity response
        """
        if self.user_context['detected_acne']:
            severity = self.user_context['severity']
            count = len(self.user_context['detected_acne'])
            
            response = f"Based on the image analysis, I detected {count} acne spot(s) with {severity} severity.\n\n"
            
            if severity == 'mild':
                response += "Good news! Mild acne typically responds well to over-the-counter treatments like benzoyl peroxide or salicylic acid."
            elif severity == 'moderate':
                response += "Moderate acne may benefit from a combination of topical treatments. Consider consulting a dermatologist for prescription options."
            else:
                response += "Severe acne often requires professional treatment. I strongly recommend seeing a dermatologist for prescription medications."
        else:
            response = "Acne severity is typically classified as:\n\n"
            response += "🟢 **Mild**: Few blackheads, whiteheads, and small pimples\n"
            response += "🟡 **Moderate**: More numerous lesions, some inflammation\n"
            response += "🔴 **Severe**: Many inflamed lesions, cysts, potential scarring\n\n"
            response += "Upload an image for a personalized assessment!"
        
        return response
    
    def _generate_cause_response(self, myths_facts: List[Dict]) -> str:
        """
        Generate response about acne causes
        
        Args:
            myths_facts: List of myths and facts
            
        Returns:
            Cause response
        """
        response = "Acne is primarily caused by four factors:\n\n"
        response += "1. **Excess oil production** - Hormones stimulate sebaceous glands\n"
        response += "2. **Clogged pores** - Dead skin cells and oil block hair follicles\n"
        response += "3. **Bacteria** - P. acnes bacteria multiply in clogged pores\n"
        response += "4. **Inflammation** - Body's immune response to bacteria\n\n"
        
        if myths_facts:
            response += "Let me clear up some common misconceptions:\n\n"
            for myth_fact in myths_facts[:2]:
                response += f"❌ **Myth**: {myth_fact['myth']}\n"
                response += f"✅ **Fact**: {myth_fact['fact']}\n\n"
        
        return response
    
    def _generate_prevention_response(self, tips: List[Dict]) -> str:
        """
        Generate prevention advice response
        
        Args:
            tips: List of lifestyle tips
            
        Returns:
            Prevention response
        """
        response = "Here are proven strategies to prevent acne breakouts:\n\n"
        
        prevention_tips = [
            "Use gentle, non-comedogenic skincare products",
            "Wash your face twice daily with a mild cleanser",
            "Avoid touching or picking at your face",
            "Change pillowcases regularly",
            "Remove makeup before bed",
            "Shower after sweating"
        ]
        
        for i, tip in enumerate(prevention_tips, 1):
            response += f"{i}. {tip}\n"
        
        if tips:
            response += "\n**Additional lifestyle factors:**\n"
            for tip in tips[:3]:
                response += f"• {tip['tip']}: {tip['description']}\n"
        
        return response
    
    def _generate_dermatologist_response(self, criteria: List[str]) -> str:
        """
        Generate dermatologist referral response
        
        Args:
            criteria: List of referral criteria
            
        Returns:
            Dermatologist response
        """
        response = "You should consider seeing a dermatologist if:\n\n"
        
        for criterion in criteria:
            response += f"• {criterion}\n"
        
        response += "\nA dermatologist can provide:\n"
        response += "- Prescription medications\n"
        response += "- Professional treatments\n"
        response += "- Personalized treatment plans\n"
        response += "- Scar prevention and treatment\n"
        
        return response
    
    def _generate_lifestyle_response(self, tips: List[Dict]) -> str:
        """
        Generate lifestyle advice response
        
        Args:
            tips: List of lifestyle tips
            
        Returns:
            Lifestyle response
        """
        response = "Lifestyle factors can significantly impact acne. Here's what you should know:\n\n"
        
        if tips:
            for tip in tips:
                response += f"**{tip['category']}**: {tip['tip']}\n"
                response += f"{tip['description']}\n\n"
        
        response += "Remember: Lifestyle changes take time to show results. Be patient and consistent!"
        
        return response
    
    def _generate_general_response(self, user_input: str, relevant_info: Dict[str, Any]) -> str:
        """
        Generate general response using retrieved information
        
        Args:
            user_input: User's message
            relevant_info: Retrieved information
            
        Returns:
            General response
        """
        if relevant_info['treatments']:
            return self._generate_treatment_response(relevant_info['treatments'])
        else:
            return random.choice(self.fallback_responses) + " Could you be more specific about what you'd like to know?"
    
    def _determine_overall_severity(self, detections: List[Dict]) -> str:
        """
        Determine overall acne severity from detections
        
        Args:
            detections: List of acne detections
            
        Returns:
            Overall severity level
        """
        if not detections:
            return 'none'
        
        severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        
        for detection in detections:
            severity = detection.get('severity', 'mild')
            severity_counts[severity] += 1
        
        # Determine overall severity
        if severity_counts['severe'] > 0:
            return 'severe'
        elif severity_counts['moderate'] > severity_counts['mild']:
            return 'moderate'
        else:
            return 'mild'
    
    def _get_fallback_response(self) -> str:
        """
        Get fallback response when other methods fail
        
        Returns:
            Fallback response
        """
        return random.choice(self.fallback_responses)
    
    def reset_conversation(self):
        """
        Reset conversation history and context
        """
        self.conversation_history = []
        self.user_context = {
            'detected_acne': None,
            'severity': None,
            'previous_treatments': [],
            'skin_type': None
        }
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of current conversation
        
        Returns:
            Conversation summary
        """
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = f"Conversation with {len(self.conversation_history)} messages.\n"
        
        if self.user_context['detected_acne']:
            count = len(self.user_context['detected_acne'])
            severity = self.user_context['severity']
            summary += f"Detected: {count} acne spots ({severity} severity)\n"
        
        return summary