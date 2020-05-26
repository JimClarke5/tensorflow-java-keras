/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class PoissonTest {
    int index;
    float epsilon = 1e-4F;
    
    public PoissonTest() {
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
     * Test of call method, of class Poisson.
     */
    @Test
    public void testConfig() {
         System.out.println("testConfig");
         Poisson instance = new Poisson();
         assertEquals("poisson", instance.getName());
         
          instance = new Poisson("poisson_loss", Reduction.SUM);
          assertEquals("poisson_loss", instance.getName());
          assertEquals( Reduction.SUM, instance.getReduction());
          
    }

   
    
    /**
     * Test of call method, of class Poisson.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            Poisson instance = new Poisson();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(tf, y_true, y_pred);
            sess.run(loss);
            float expected = -3.306581945521002F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    /**
     * Test of call method, of class Poisson.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            Poisson instance = new Poisson();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = -7.605138474698304F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            Poisson instance = new Poisson();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2,1)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = -6.147338926788071F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_zero_weighted() {
        System.out.println("test_zero_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
            Poisson instance = new Poisson();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand sampleWeight =  tf.constant(0.F);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
    @Test
    public void test_timestep_weighted() {
        System.out.println("test_timestep_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            Poisson instance = new Poisson(Reduction.AUTO);
              float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3,1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3,1)));
            Operand sampleWeight =  tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2,3)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            
            sess.run(loss);
            float expected = -12.263126013890561F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            assertEquals(expected, f.getFloat(), epsilon);
                         });
            }
        }
    }
    
}
