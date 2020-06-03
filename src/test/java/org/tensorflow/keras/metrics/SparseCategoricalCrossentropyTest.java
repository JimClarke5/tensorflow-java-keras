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
public class SparseCategoricalCrossentropyTest {
    
     private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public SparseCategoricalCrossentropyTest() {
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
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void testConfig() {
        System.out.println("testConfig");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf,"scce", true, 0.2F, -1, TInt32.DTYPE);
            assertEquals("scce", instance.getName());
            assertEquals(0.2F, instance.getLabelSmoothing());
            assertTrue(instance.isFromLogits());
            assertEquals(TInt32.DTYPE, instance.getDataType());
        }
    }
    
    @Test
    public void testUnweighted() {
        System.out.println("testUnweighted");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 2};
            float[] pred_np = {0.05f, 0.95f, 0f, 0.1f, 0.8f, 0.1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(2.3538785F, total);
            session.evaluate(2, count);
            session.evaluate(1.1769392F, result);
            
        }
    }
    
    @Test
    public void test_unweighted_with_logits() {
        System.out.println("test_unweighted_with_logits");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {1, 2};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(7.002277F, total);
            session.evaluate(2, count);
            session.evaluate(3.501135F, result);
            
        }
    }
    
    @Test
    public void test_weighted() {
        System.out.println("test_weighted");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            session.run(instance.resetStates());
            int[] true_np = {1, 2};
            float[] pred_np = {0.05f, 0.95f, 0f, 0.1f, 0.8f, 0.1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5F, 2.F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(4.6821103F, total);
            session.evaluate(3.5, count);
            session.evaluate(1.3377458F, result);
            
        }
    }
    
    @Test
    public void test_weighted_from_logits() {
        System.out.println("test_weighted_from_logits");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true);
            session.run(instance.resetStates());
            int[] true_np = {1, 2};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand sampleWeight = tf.constant(new float[] {1.5F, 2F});
            Op op = instance.updateState(y_true, y_pred, sampleWeight);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(14.004333F, total);
            session.evaluate(3.5f, count);
            session.evaluate(4.001232F, result);
            
        }
    }
    
    @Test
    public void test_axis() {
        System.out.println("test_axis");
        try(TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            int axis = 0;
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, axis);
            session.run(instance.resetStates());
            int[] true_np = {1, 2};
            float[] pred_np = {0.05f, 0.1f, 0.95f, 0.8f, 0f, 0.1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 2)));
            Op op = instance.updateState(y_true, y_pred);
            session.run(op);
            Variable<TFloat32> total = instance.getVariable(TOTAL);
            Variable<TInt32> count = instance.getVariable(COUNT);
            Operand result  = instance.result();
            session.evaluate(2.3538785F, total);
            session.evaluate(2, count);
            session.evaluate(1.1769392F, result);
            
        }
    }
    
}
