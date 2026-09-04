"""Deterministic regression checks; no model downloads or API calls."""
import unittest
from unittest.mock import Mock, patch
from src.rag import SupportRAG

class RoutingChecks(unittest.TestCase):
    def engine(self, review=False, label='sad'):
        engine=SupportRAG.__new__(SupportRAG)
        engine.language=Mock(); engine.language.predict.return_value={'language':'en','language_name':'English'}
        engine.intent=Mock(); engine.intent.predict.return_value={'candidate_intent':'track_order','priority':False,'margin':1,'route':'order_status'}
        engine.emotion=Mock(); engine.emotion.predict.return_value={'label':label,'confidence':0.7 if review else 0.97,'needs_review':review}
        engine.retrieve=Mock(return_value=[{'id':'doc1','instruction':'Track order','response':'Check the store website.','intent':'track_order','score':0.8}])
        return engine

    def run_answer(self,engine,draft=None,safe=True,history=None,prepared=None):
        draft=draft or {'answer':'Check the store website.','source_ids':['1'],'status':'answered'}
        calls=[]
        def complete(client,system,payload,schema,name):
            calls.append((name,payload))
            if name=='search_query':return prepared
            if name=='grounded_answer':return draft
            return {'violations':[] if safe else ['action_promise'],'fallback':'I cannot verify that. Please check the store’s official support.'}
        with patch('src.rag.complete_json',side_effect=complete):
            result=engine.answer('Where is my order?',history=history,api_key='test-key')
        return result,calls

    def test_uncertain_emotion_does_not_set_priority_or_assert_sadness(self):
        result,calls=self.run_answer(self.engine(review=True))
        self.assertEqual(result['sentiment']['label'],'uncertain')
        self.assertFalse(result['priority'])
        self.assertEqual(calls[0][1]['sentiment'],'uncertain')

    def test_confident_negative_sets_priority_without_complaint(self):
        result,_=self.run_answer(self.engine())
        self.assertTrue(result['priority'])

    def test_followup_does_not_replace_english_emotional_wording(self):
        engine=self.engine()
        self.run_answer(engine,history=[{'role':'user','content':'Earlier question'}],prepared={'english_query':'Track a late order','emotion_text':'rewritten','reply_language':'English'})
        engine.emotion.predict.assert_called_once_with('Where is my order?')
        engine.retrieve.assert_called_once_with('Track a late order',intent='track_order')

    def test_translation_separates_emotion_from_query(self):
        engine=self.engine();engine.language.predict.return_value={'language':'ar','language_name':'Arabic'}
        self.run_answer(engine,prepared={'english_query':'Track order','emotion_text':'I am furious!','reply_language':'Arabic'})
        engine.emotion.predict.assert_called_once_with('I am furious!')

    def test_feedback_without_sources_is_valid(self):
        result,_=self.run_answer(self.engine(label='happy'),draft={'answer':'Thank you for sharing your feedback.','source_ids':[],'status':'conversation'})
        self.assertEqual(result['status'],'conversation');self.assertEqual(result['sources'],[])

    def test_unsupported_promise_or_bad_citation_falls_back(self):
        for ids,safe in [(['1'],False),(['999'],True),([],True)]:
            with self.subTest(ids=ids,safe=safe):
                result,_=self.run_answer(self.engine(),draft={'answer':'I will connect you.','source_ids':ids,'status':'answered'},safe=safe)
                self.assertEqual(result['status'],'not_supported');self.assertEqual(result['sources'],[])
                self.assertNotIn('connect you',result['answer'])

    def test_positive_review_acknowledges_without_generation(self):
        engine=self.engine(label='happy')
        engine.intent.predict.return_value={'candidate_intent':'review','priority':True,'margin':1,'route':'complaint'}
        with patch('src.rag.complete_json') as complete:
            result=engine.answer('My order arrived on time and the quality is amazing. I just wanted to leave a positive review.',api_key='test-key')
        self.assertEqual(result['status'],'conversation')
        self.assertIn('haven’t submitted',result['answer'])
        engine.retrieve.assert_not_called();complete.assert_not_called()

    def test_explicit_human_request_has_no_fake_transfer(self):
        engine=self.engine()
        result=engine.answer('Please connect me to a human agent.')
        self.assertEqual(result['status'],'human_requested')
        self.assertIn('can’t connect',result['answer'])
        engine.retrieve.assert_not_called()

    def test_social_messages_bypass_retrieval_and_api(self):
        engine=self.engine()
        with patch('src.rag.complete_json') as complete:
            for message in ['hey','good morning','thank you very much','goodbye']:
                self.assertEqual(engine.answer(message)['status'],'conversation')
        engine.retrieve.assert_not_called();complete.assert_not_called()

if __name__=='__main__':unittest.main()
