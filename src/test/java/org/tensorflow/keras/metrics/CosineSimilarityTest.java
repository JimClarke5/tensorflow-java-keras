/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import static org.tensorflow.keras.metrics.impl.Reduce.COUNT;
import static org.tensorflow.keras.metrics.impl.Reduce.TOTAL;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class CosineSimilarityTest {
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public CosineSimilarityTest() {
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

      /**
     * Test of call method, of class CosineSimilarity.
     */
   @Test
    public void testConfig() {
        System.out.println("testConfig");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf,"cosine", TInt32.DTYPE);
            assertEquals("cosine", instance.getName());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        System.out.println("testUnweighted");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Operand<TFloat32> loss = instance.call(y_true, y_pred, null);
            session.print(System.out, loss);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(0.3744381F, total);
            session.evaluate(2, count);
            session.evaluate(0.18721905F, result);
            
        }
    }
    
    
    @Test
    public void test_weighted() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.2f, 3.4f});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(-0.3119840621948241F, total);
            session.evaluate(4.6, count);
            session.evaluate(-0.06782262221626612F, result);
            
        }
    }
    
    @Test
    public void test_axis() {
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            int axis = 1;
            CosineSimilarity instance = new CosineSimilarity(tf, axis);
            session.run(instance.resetStates());
            float[] true_np = { 1, 9, 2, -5, -2, 6 };
            float[] pred_np = { 4, 8, 12, 8, 1, 3 };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(0.3744381F, total);
            session.evaluate(2, count);
            session.evaluate(0.18721905F, result);
            
        }
    }
}
