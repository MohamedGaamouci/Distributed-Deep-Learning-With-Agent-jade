package com.jade.ddl.Agents;

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.HashSet;
import java.util.Set;

public class ModelAggregatorAgent extends Agent {
    private final Set<String> receivedModels = new HashSet<>();
    private int numWorkers = 3; // Number of workers expected to send models

    @Override
    protected void setup() {
        Object[] args = getArguments();
        if (args != null && args.length > 0) {
            // Cast the argument back to Integer and then to int
            this.numWorkers = (int) args[0];
        }

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = receive();
                if (msg != null) {
                    String content = msg.getContent();
                    if(content.startsWith("Model from")){
                        handleModelReception(content);
                    } else if (content.equals("Prepare for Aggregation")) {
                        handlePreparationRequest(msg);
                        receivedModels.clear();
                    }
                } else {
                    block();
                }
            }
        });
    }

    /**
     * Handles preparation request from CoordinatorAgent.
     */
    private void handlePreparationRequest(ACLMessage msg) {
        System.out.println(getLocalName() + " is preparing for aggregation.");
        ACLMessage reply = msg.createReply();
        reply.setPerformative(ACLMessage.INFORM);
        reply.setContent("Aggregator Ready");
        send(reply);
    }

    /**
     * Handles the reception of a model from a worker.
     *
     * @param content The content of the received message.
     */
    private void handleModelReception(String content) {
        receivedModels.add(content);
        System.out.println(getLocalName() + " received: " + content);

        // Check if all models are received
        if (receivedModels.size() == numWorkers) {
            System.out.println(getLocalName() + " has received all models. Starting aggregation.");

            // Proceed aggregation
            performAggregation();

        }
    }

    /**
     * Simulates the model aggregation process.
     */
    private void performAggregation() {
        try {
            // Set up the connection
            URL url = new URL("http://localhost:5000/aggregate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setDoOutput(true);


            // Read the server's response
            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                System.out.println("Aggregation Complete successfully!");
                notifyAggregationComplete();
            } else {
                System.err.println("Server {Aggregation} responded with code: " + responseCode);
            }
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (ProtocolException e) {
            throw new RuntimeException(e);
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println(getLocalName() + " has completed model aggregation.");
    }

    /**
     * Notifies the CoordinatorAgent that aggregation is complete.
     */
    private void notifyAggregationComplete() {
        ACLMessage aggregationCompleteMsg = new ACLMessage(ACLMessage.INFORM);
        aggregationCompleteMsg.addReceiver(new AID("Coordinator", AID.ISLOCALNAME));
        aggregationCompleteMsg.setContent("Aggregation Complete.");
        send(aggregationCompleteMsg);
    }
}
