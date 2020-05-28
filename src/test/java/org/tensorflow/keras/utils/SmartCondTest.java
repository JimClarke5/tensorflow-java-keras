/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.util.function.Supplier;
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
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.TBool;

/**
 *
 * @author Jim Clarke
 */
public class SmartCondTest {
    
    public SmartCondTest() {
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
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondOp() {
        System.out.println("testCondOp");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            Constant pred = tf.constant(true);
            sess.run(pred);
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () -> tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(tf, pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
    @Test
    public void testCondOp2() {
        System.out.println("testCondOp2");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            Constant pred = tf.constant(true);
            sess.run(pred);
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(tf, pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
    @Test
    public void testCondOpPoint1() {
        System.out.println("testCondOpPoint1");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            Operand pred = tf.math.equal(tf.constant(0.1), tf.constant(1.0));
            sess.run(pred);
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = false;
            Operand resultOp = SmartCond.cond(tf, pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
        }
    }
    
    @Test
    public void testCondOpString() {
        System.out.println("testCondOpString");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            Operand pred = tf.math.equal(tf.constant("TRUE"), tf.constant("TRUE"));
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(tf, pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
        }
    }
    
    /**
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondBoolean() {
        System.out.println("testCondBoolean");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            boolean pred = false;
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = false;
            Operand resultOp = SmartCond.cond( pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
    /**
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondInt() {
        System.out.println("testCondInt");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            int pred = 1;
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
    /**
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondFloat1() {
        System.out.println("testCondFloat1");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            float pred = 1.0F;
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    /**
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondFloat0_1() {
        System.out.println("testCondFloat0_1");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            float pred = 0.1F;
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = false;
            Operand resultOp = SmartCond.cond(pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
    /**
     * Test of cond method, of class SmartCond.
     */
    @Test
    public void testCondString() {
        System.out.println("testCondString");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            String pred = "true";
            
            Supplier<Operand> true_fn = () -> tf.constant(true);
            Supplier<Operand> false_fn = () ->  tf.constant(false);
            boolean expResult = true;
            Operand resultOp = SmartCond.cond(pred, true_fn, false_fn);
            boolean actualResult;
            try (Tensor<TBool> result = sess.runner().fetch(resultOp).run().get(0).expect(TBool.DTYPE)) {
                   actualResult = result.data().getBoolean();
                   assertEquals(expResult, actualResult);
             }
            
        }
    }
    
}
