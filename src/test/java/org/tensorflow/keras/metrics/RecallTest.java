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

import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.metrics.impl.MetricsImpl;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

/**
 *
 * @author jbclarke
 */
public class RecallTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public RecallTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
        Metrics.setDebug(null);
    }
    
    @AfterEach
    public void tearDown() {
    }

    @Test
    public void testConfig() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf,"my_recall", new float[]{0.4f, 0.9f}, 15, 12);
            session.run(instance.initializeVars());
            
            assertEquals("my_recall", instance.getName());
            assertEquals(2, instance.getVariables().size());
            Float[] expected = new Float[] {0f, 0f };
            session.evaluate(expected, instance.getTruePositives());
            session.evaluate(expected, instance.getFalseNegatives());
            assertArrayEquals(new float[]{0.4f, 0.9f}, instance.getThresholds());
            assertEquals(15, instance.getTopK());
            assertEquals(12, instance.getClassID());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_value_is_idempotent() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf,new float[]{0.3f, 0.72f});
            session.run(instance.initializeVars());
            
            Operand yPred = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE);
            Operand yTrue = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE);
            Op update = instance.updateState(yTrue,yPred );
            
            for(int i = 0; i < 10; i++)
                session.run(update);
            
            Operand initial_recall = instance.result();
            for(int i = 0; i < 10; i++)
                session.evaluate(initial_recall, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_unweighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf);
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{1, 0, 1, 0}});
            Operand yTrue = tf.constant(new float[][] {{0, 1, 1, 0}});
            Op update = instance.updateState(yTrue,yPred );
            session.run(update);
            
            session.evaluate(0.5f, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    private Random random = new Random();
    
    private int[][] generateRandomArray(int dim1, int dim2, int maxInt) {
        int[][] result = new int[dim1][dim2];
        for(int i = 0; i < dim1; i++) {
            for(int j = 0; j < dim2; j++) {
                result[i][j] = random.nextInt(2);
            }
        }
        
        return result;
    }
    
    @Test
    public void test_unweighted_all_incorrect() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf);
            session.run(instance.initializeVars());
            int[][] array = generateRandomArray(100, 1, 2);
            Operand yPred = tf.dtypes.cast(tf.constant(array), TFloat32.DTYPE);
            Operand yTrue = tf.dtypes.cast(tf.math.sub(tf.constant(1), tf.constant(array)), TFloat32.DTYPE);
            Op update = instance.updateState(yTrue,yPred );
            session.run(update);
            
            session.evaluate(0.f, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[][] {
                {1, 0, 1, 0},
                {0, 1, 0, 1}
            });
            Operand yTrue = tf.constant(new float[][] {
                {0, 1, 1, 0},
                {1, 0, 0, 1}
            });
            
            Operand sampleWeight = tf.constant(new float[][] {
                {1, 2, 3, 4},
                {4, 3, 2, 1}
            });
            Op update = instance.updateState(yTrue,yPred );
            session.run(update);
            
            float weighted_tp = 3.0f + 1.0f;
            float weighted_t = (2.0f + 3.0f) + (4.0f + 1.0f);
            float expected_recall = weighted_tp / weighted_t;
            
            session.evaluate(expected_recall, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_div_by_zero() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf);
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[] { 0, 0, 0, 0});
            Operand yTrue = tf.constant(new float[] { 0, 0, 0, 0});
            
            Op update = instance.updateState(yTrue,yPred );
            session.run(update);
            
            session.evaluate(0f, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_unweighted_with_threshold() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, new float[]{0.5f, 0.7f});
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{ 1, 0, 0.6f, 0}});
            Operand yTrue = tf.constant(new float[][] {{ 0, 1, 1, 0}});
            
            Op update = instance.updateState(yTrue,yPred );
            session.run(update);
            
            Float[] expected = new Float[]{0.5f, 0f};
            session.evaluate(expected, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void test_weighted_with_threshold() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, new float[]{0.5f, 1.f});
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[][] {{ 0, 1}, {1, 0}});
            Operand yPred = tf.constant(new float[][] {{ 1, 0}, {0.6f, 0}});
            Operand weights = tf.constant(new float[][] {{ 1, 4}, {3, 2}});
            
            Op update = instance.updateState(yTrue,yPred, weights);
            session.run(update);
            
            float weighted_tp = 0 + 3.f;
            float weighted_positives = (0 + 3.f) + (4.f + 0.f);
            float expected_recall = weighted_tp / weighted_positives;
            Float[] expected = new Float[] {expected_recall, 0f};
            session.evaluate(expected, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void  test_multiple_updates() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, new float[]{0.5f, 1.f});
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[][] {{ 0, 1}, {1, 0}});
            Operand yPred = tf.constant(new float[][] {{ 1, 0}, {0.6f, 0}});
            Operand weights = tf.constant(new float[][] {{ 1, 4}, {3, 2}});
            
            Op update = instance.updateState(yTrue,yPred, weights);
            for(int i = 0; i < 2; i++)
                session.run(update);
            
            float weighted_tp = (0f + 3.f) + (0f + 3.f);
            float weighted_positives = ((0f + 3.f) + (4.f + 0.f)) + ((0f + 3.f) + (4.f + 0.f));
            float expected_recall = weighted_tp / weighted_positives;
            Float[] expected = new Float[] {expected_recall, 0f};
            session.evaluate(expected, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void  test_unweighted_top_k() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, null, null, 3, Recall.NO_CLASSID);
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[][] {{0f, 1f, 1f, 0f, 0f}});
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.5f, 0f, 0.2f}});
            
            Op update = instance.updateState(yTrue,yPred);
            session.run(update);
            
            session.evaluate(0.5f, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    @Test
    public void  test_weighted_top_k() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf,  3, Recall.NO_CLASSID);
            instance.resetDebug();
            session.run(instance.initializeVars());
            
            Operand yTrue = tf.constant(new float[][] {{0, 1, 1, 0, 1}});
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.4f, 0f, 0.2f}});
            Operand weights = tf.constant(new float[][] {{1, 4, 2, 3, 5}});
            
            Op update = instance.updateState(yTrue,yPred, weights);
            session.run(update);
            
            yTrue = tf.constant(new float[][] {{1, 0, 1, 1, 1}});
            yPred = tf.constant(new float[][] {{0.2f, 0.6f, 0.4f, 0.2f, 0.2f}});
            weights = tf.constant(3.f);
            
            update = instance.updateState(yTrue,yPred, weights);
            session.run(update);
                    
            float weighted_tp = (2 + 5) + (3 + 3);
            float weighted_positives = (4 + 2 + 5) + (3 + 3 + 3 + 3);
            float expected_recall = weighted_tp / weighted_positives;
            session.evaluate(expected_recall, instance.result());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void  test_unweighted_class_id() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, Recall.NO_TOPK, 2);
            instance.resetDebug();
            session.run(instance.initializeVars());
            
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.6f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new float[][] {{0, 1, 1, 0, 0}});
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            System.out.println("++++++++++++++++  TRUE_POSITIVES ++++++++++++++");
            session.print(System.out, instance.getTruePositives());
            System.out.println("++++++++++++++++  FALSE_NEGATIVES ++++++++++++++");
            session.print(System.out, instance.getFalseNegatives());
            session.evaluate(1f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(0f, instance.getFalseNegatives());
            
            yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0f, 0f, 0.2f}});
            yTrue = tf.constant(new float[][] {{0, 1, 1, 0, 0}});
            update = instance.updateState(yTrue,yPred);
            session.run(update);
            session.evaluate(0.5f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(1f, instance.getFalseNegatives());
            
            yPred = tf.constant(new float[][] {{0.2f, 0.1f, 0.6f, 0f, 0.2f}});
            yTrue = tf.constant(new float[][] {{0, 1, 0, 0, 0}});
            update = instance.updateState(yTrue,yPred);
            session.run(update);
            session.evaluate(0.5f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(1f, instance.getFalseNegatives());
            
            
        } finally {
            Metrics.resetDebug();
        }
    }
    
    @Test
    public void  test_unweighted_top_k_and_class_id() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf,  2, 2);
            instance.resetDebug();
            session.run(instance.initializeVars());
            
           
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.6f, 0.3f, 0, 0.2f}});
            Operand yTrue = tf.constant(new float[][] {{0, 1, 1, 0, 0}});
            Op update = instance.updateState(yTrue,yPred);
            session.run(update);
            
            session.evaluate(1f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(0f, instance.getFalseNegatives());
            
            
            yPred = tf.constant(new float[][] {{1, 1, 0.9f, 1, 1}});
            yTrue = tf.constant(new float[][] {{0, 1, 1, 0, 0}});
            
            update = instance.updateState(yTrue,yPred);
            session.run(update);
            session.evaluate(0.5f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(1f, instance.getFalseNegatives());
        } finally {
            Metrics.resetDebug();
        }
    }
    
    
    @Test
    public void  test_unweighted_top_k_and_threshold() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Recall instance = new Recall(tf, 0.7f, 2, Recall.NO_CLASSID);
            session.run(instance.initializeVars());
            
           
            Operand yPred = tf.constant(new float[][] {{0.2f, 0.8f, 0.6f, 0f, 0.2f}});
            Operand yTrue = tf.constant(new float[][] {{1, 1, 1, 0, 1}});
            Op update = instance.updateState(yTrue,yPred);
            session.run(update);
            
            session.evaluate(0.25f, instance.result());
            session.evaluate(1f, instance.getTruePositives());
            session.evaluate(3f, instance.getFalseNegatives());
            
        } finally {
            Metrics.resetDebug();
        }
    }
    
}
