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
import org.tensorflow.op.random.RandomUniform;
import org.tensorflow.op.random.RandomUniformInt;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class PrecisionAtRecallTest {
    
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public PrecisionAtRecallTest() {
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
            PrecisionAtRecall instance = new PrecisionAtRecall(tf,"precision_at_recall_1", 0.4f, 100 );
            assertEquals("precision_at_recall_1", instance.getName());
            assertEquals(0.4f, instance.getRecall());
            assertEquals(100, instance.getNumThresholds());
            assertEquals(4, instance.getVariables().size());
        }
    }
    
     @Test
    public void test_value_is_idempotent() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.7f);
            session.run(instance.initializeVars());
            
            Operand yPred = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE, RandomUniform.seed(1L));
            Operand yTrue = tf.random.randomUniform(tf.constant(Shape.of(10,3)), TFloat32.DTYPE, RandomUniform.seed(1L));
            
            //instance.setDebug(session.getGraphSession());
            Op update = instance.updateState(yTrue, yPred);
            
            for(int i = 0; i < 10; i++)
                session.run(update);
            
            Operand initialPrecision = instance.result();
            
            for(int i = 0; i < 10; i++)
                session.evaluate(initialPrecision, instance.result());
            
            //instance.setDebug(null);
                
            
        }
    }
    
    private Random random = new Random();
    
    private int[][] generateRandomArray(int dim1, int dim2) {
        int[][] result = new int[dim1][dim2];
        for(int i = 0; i < dim1; i++) {
            for(int j = 0; j < dim2; j++) {
                result[i][j] = random.nextInt(2);
            }
        }
        
        return result;
    }
    
    @Test
    public void  test_unweighted_all_correct() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.7f);
            session.run(instance.initializeVars());
            int[][] predArray = generateRandomArray(100,1); 
            int[][] trueArray = new int[100][1]; // 100,1
            System.arraycopy(predArray, 0, trueArray, 0, predArray.length);
            Operand yPred = tf.constant(predArray);
            Operand yTrue = tf.constant(trueArray);
            yPred = tf.dtypes.cast(yPred, TFloat32.DTYPE);
            yTrue = tf.dtypes.cast(yTrue, TFloat32.DTYPE);
            
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            
            Operand precision = instance.result();
            
            session.evaluate(1f, precision);
        }
    }
    
    @Test
    public void  test_unweighted_high_recall() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.8f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[] { 
                0.0f, 0.1f, 0.2f, 0.3f, 0.5f, 0.4f, 0.5f, 0.6f, 0.8f, 0.9f} );
            Operand yTrue = tf.constant(new long[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
            
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            
            Operand precision = instance.result();
            
            session.evaluate(0.8f, precision);
        }
    }
    
    @Test
    public void  test_unweighted_low_recall() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.4f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[] { 
                0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.15f, 0.25f, 0.26f, 0.26f} );
            Operand yTrue = tf.constant(new long[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
            
            Op update = instance.updateState(yTrue, yPred);
            session.run(update);
            
            Operand precision = instance.result();
            
            session.evaluate(0.5f, precision);
        }
    }
    
    public void   test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.4f);
            session.run(instance.initializeVars());
            Operand yPred = tf.constant(new float[] { 
                0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.01f, 0.02f, 0.25f, 0.26f, 0.26f} );
            Operand yTrue = tf.constant(new long[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
            Operand sampleWeight = tf.constant(new float[] {2, 2, 1, 1, 1, 1, 1, 2, 2, 2});
            
            Op update = instance.updateState(yTrue, yPred, sampleWeight);
            session.run(update);
            
            Operand precision = instance.result();
            
            session.evaluate(2.f/3.f, precision);
        }
    }
    
    public void  test_invalid_sensitivity() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, -1f);
            fail();
        }catch(AssertionError expected) {
            
        }
    }
    
    public void  test_invalid_num_thresholds() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            PrecisionAtRecall instance = new PrecisionAtRecall(tf, 0.7f, -1);
            fail();
        }catch(AssertionError expected) {
            
        }
    }
    
}
