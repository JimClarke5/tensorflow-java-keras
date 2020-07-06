/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

/**
 *
 * @author jbclarke
 */
public class AUCTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    float epsilon = 1e-4F;
     
    int numThresholds = 3;
    float[] pred_array = new float[]{0f, 0.5f, 0.3f, 0.9f};
    int[] true_array = new int[] {0, 0, 1, 1};
    float[] sampleWeight= new float[]{1, 2, 3, 4};
    
    public AUCTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
        
    }
    
    @AfterEach
    public void tearDown() {
    }
    
    
    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            AUC instance = new AUC(tf,"area_under_curve", numThresholds, AUCCurve.ROC, AUCSummationMethod.MAJORING);
            session.run(tf.init());
             instance.setDebug(null);
            assertEquals("area_under_curve", instance.getName());
            assertEquals(numThresholds, instance.getNumThresholds());
           
            assertEquals(AUCCurve.ROC, instance.getCurve());
            assertEquals(AUCSummationMethod.MAJORING, instance.getSummationMethod());
            
        }
    }

   
    
    
    @Test
    public void test_config_manual_thresholds() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            
            float[] thresholds = new float[]{0.3f, 0.5f};
            AUC instance = new AUC(tf,"auc1", thresholds, AUCCurve.PR, AUCSummationMethod.MINORING);
            session.run(tf.init());
            assertEquals("auc1", instance.getName());
            float[] expectedThresholds = new float[] {0.0f, 0.3f, 0.5f, 1.0f};
            assertArrayEquals(expectedThresholds, instance.getThresholds(),epsilon);
            assertEquals(AUCCurve.PR, instance.getCurve());
            assertEquals(AUCSummationMethod.MINORING, instance.getSummationMethod());
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            instance.updateState(yTrue, yPred );
            assertEquals(4, instance.getVariables().size());
            
        }
    }
    
    
    
    @Test
    public void test_value_is_idempotent() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            AUC instance = new AUC(tf, numThresholds);
            
            
            session.run(tf.init());
            
            Op update = instance.updateState(yTrue, yPred );
            
            for(int i = 0; i < 10; i++) {
                session.run(update);
            }
            
            Operand result = instance.result();
            
             for(int i = 0; i < 10; i++) {
                 session.evaluate(result, instance.result());
            }
        }
    }
    
    @Test
    public void basic_test() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
             AUC instance = new AUC(tf);
             assertEquals(numThresholds, instance.getNumThresholds());
             float[] expectedThresholds = new float[] {0.0f,  0.5f, 1 + 1e-7f};
             assertArrayEquals(expectedThresholds, instance.getThresholds(), epsilon);
             
             instance.resetStates();
             Operand yPred = tf.constant(new float[] {0, 0, 1, 1});
             Operand yTrue = tf.constant(new float[] {0f, 0.5f, 0.3f, 0.9f});
             
             Op update = instance.updateState(yTrue,yPred);
             session.run(update);
             Operand result = instance.result();
             session.evaluate(0.75, instance.result()); 
              
        }
    }
    
    @Test
    public void basic_test_sample_weight() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
             AUC instance = new AUC(tf, numThresholds);
             assertEquals(numThresholds, instance.getNumThresholds());
             float[] expectedThresholds = new float[] {-1e07f,  0.5f, 1 + 1e-7f};
             assertArrayEquals(expectedThresholds, instance.getThresholds(), epsilon);
             
             instance.resetStates();
             Operand yPred = tf.constant(new float[] {0, 0, 1, 1});
             Operand yTrue = tf.constant(new float[] {0f, 0.5f, 0.3f, 0.9f});
             Operand sampleWeights = tf.constant(new float[] {1, 0, 0, 1});
             
             Op update = instance.updateState(yTrue,yPred, sampleWeights);
             session.run(update);
             Operand result = instance.result();
             session.evaluate(1.0f, instance.result()); 
              
        }
    }
    
    @Test
    public void test_unweighted_all_correct() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            AUC instance = new AUC(tf, this.numThresholds);
            session.run(tf.init());
            
            Op update = instance.updateState(yTrue, yTrue );
            session.run(update);
            Operand result = instance.result();
            
            session.evaluate(1, instance.result());
        }
    }
    @Test
    public void test_unweighted() {
         try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            AUC instance = new AUC(tf, this.numThresholds);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            
            //float expected_result = (0.75f * 1 + 0.25f * 0);
            session.evaluate(0.75f, instance.result());
         }
    }

    @Test
    public void test_manual_thresholds() {
         try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            AUC instance = new AUC(tf, new float[] { 0.5f });
            float[] expectedThresholds = new float[] {-1e-7f,  0.5f, 1 + 1e-7f};
             assertArrayEquals(expectedThresholds, instance.getThresholds(), epsilon);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred );
            session.run(update);
            Operand result = instance.result();
            
            //float expected_result = (0.75f * 1 + 0.25f * 0);
            session.evaluate(0.75f, instance.result());
         }
    }
    
    @Test
    public void test_weighted_roc_interpolation() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result = (0.78571427f * 1 + 0.2857145f * 0);
            session.evaluate(expected_result, instance.result()); 
         }
    }
    
    @Test
    public void test_weighted_roc_majoring() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds, AUCCurve.ROC, AUCSummationMethod.MAJORING);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result  = (1f * 1f + .5714285f * 0f);
            session.evaluate(expected_result, instance.result()); 
         }
    }
    
    @Test
    public void  test_weighted_roc_minoring() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds, AUCCurve.ROC, AUCSummationMethod.MINORING);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            
            float expected_result  = (1f * 0.5714285f + 0f*  0f);
            session.evaluate(expected_result, instance.result()); 
         }
    }
    
    @Test
    public void test_weighted_pr_majoring() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds, AUCCurve.PR, AUCSummationMethod.MAJORING);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            float expected_result  = 1f * 0.4285715f + 1f * 0.5714285f;
            session.evaluate(expected_result, instance.result()); 
        }
    }
    
    @Test
    public void test_weighted_pr_minoring() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds, AUCCurve.PR, AUCSummationMethod.MINORING);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            float expected_result  = 0.7f * 0.4285715f + 0f * 0.5714285f;
            session.evaluate(expected_result, instance.result()); 
        }
    }
    
    @Test
    public void test_weighted_pr_interpolation() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, this.numThresholds, AUCCurve.PR);
            session.run(tf.init());
            Op update = instance.updateState(yTrue, yPred, sampleWights );
            session.run(update);
            Operand result = instance.result();
            float expected_result  = 0.916613f;
            session.evaluate(expected_result, instance.result()); 
        }
    }
    
    @Test
    public void test_invalid_num_thresholds() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Operand yPred = tf.constant(this.pred_array);
            Operand yTrue = tf.constant(this.true_array);
            Operand sampleWights = tf.constant(this.sampleWeight);
            
            AUC instance = new AUC(tf, -1);
            fail();
        }catch (AssertionError expected) {
            
        }
    }
    
    @Test
    public void test_extra_dims() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            // logits = scipy.special.expit(-np.array([[[-10., 10., -10.], [10., -10., 10.]],
            //                   [[-12., 12., -12.], [12., -12., 12.]]],
            //                  dtype=np.float32))
            float[][][] logitsArray = {
                {{9.99954602e-01f, 4.53978687e-05f, 9.99954602e-01f},
                            {4.53978687e-05f, 9.99954602e-01f, 4.53978687e-05f}},
                {{9.99993856e-01f, 6.14417460e-06f, 9.99993856e-01f},
                            {6.14417460e-06f, 9.99993856e-01f, 6.14417460e-06f}}
            };
            
            long[][][] labelArray = {
                {{1, 0, 0}, {1, 0, 0}},
                {{0, 1, 1}, {0, 1, 1}}
            };
                    
                    
            Operand logits = tf.constant(logitsArray);
            Operand labels = tf.constant(labelArray);
            
            AUC instance = new AUC(tf);
            session.run(tf.init());
            Op update = instance.updateState(labels, logits );
            session.run(update);
            Operand result = instance.result();
            float expected_result  = 0.5f;
            session.evaluate(expected_result, instance.result()); 
        }
    }
}
