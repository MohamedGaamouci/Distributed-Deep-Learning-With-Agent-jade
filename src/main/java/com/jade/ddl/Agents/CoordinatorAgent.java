package com.jade.ddl.Agents;



import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.UnreadableException;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CoordinatorAgent extends Agent {
    private int numWorkers = 3; // Number of workers
    private final Set<String> readyWorkers = new HashSet<>();
    private final Set<String> completedWorkers = new HashSet<>();

    private int round = 0;
    private String trainingConfig = "{"
            + "\"optimizer\": {"
            + "\"type\": \"Adam\","
            + "\"learning_rate\": 0.001"
            + "},"
            + "\"loss_function\": \"CrossEntropyLoss\","
            + "\"training\": {"
            + "\"epochs\": 1,"
            + "\"batch_size\": 32"
            + "}"
            + "}";

    @Override
    protected void setup() {
        // Define the model architecture and training config

        Object[] args = getArguments();
        if (args != null && args.length > 0) {
            // Cast the argument back to Integer and then to int
            this.numWorkers = (int) args[0];
        }

        System.out.println(getLocalName() + " is ready.");

        // Start workflow by requesting data split
        requestDataSplit();

        // Handle responses
        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = receive();
                if (round == 1){
                    msg = null;
                }
                if (msg != null ) {
                    String sender = msg.getSender().getLocalName();
                    String content = msg.getContent();

                    System.out.println(getLocalName() + " received: <<" + content + ">>' from " + sender);

                    if (content.equals("Data Split Complete")) {
                        // Step 2: Notify workers to prepare for training
                        notifyWorkersToPrepare();
                    } else if (content.equals("Ready for Training")) {
                        // Step 3: Track ready workers
                        readyWorkers.add(sender);
                        // Send the Model Configuration
                        ACLMessage reply = msg.createReply();
                        reply.setPerformative(ACLMessage.INFORM);
                        reply.setContent(trainingConfig);
                        send(reply);

                    } else if (content.equals("Training Complete.")) {
                        // Step 5: Track completed workers
                        completedWorkers.add(sender);
                        if (completedWorkers.size() == numWorkers) {
                            // Step 6: Notify ModelAggregatorAgent to prepare
                            System.out.println("---------- Training Complete --------");
                            notifyAggregatorToPrepare();
                        }
                    } else if (content.equals("Aggregator Ready")) {
                        // Step 7: Notify workers to send their models
                        System.out.println("---- Aggregating Begin ... ----");
                        notifyWorkersToSendModels();
                    } else if (content.equals("Aggregation Complete.")) {
                        // Final step: Log aggregation completion
                        System.out.println("Coordinator: Aggregation process completed!");

                        ACLMessage msg2 = new ACLMessage(ACLMessage.REQUEST);
                        msg2.addReceiver(new AID("Test", AID.ISLOCALNAME));
                        msg2.setContent("Test The Model");
                        send(msg2);

                    }else if (content.equals("Good")){
                            // Deserialize the map
                        System.out.println("Coordinate receive from Test agent ::");
                        System.out.println("Coordinator: < Testing Complete > .");
                        System.out.println("MAS DLL Complete .");
                        round =1;
                    } else if (content.equals("Train again")) {
                        System.out.println(getLocalName() + ":: <<Training Again>>");
                        completedWorkers.clear();
                        for (int i = 1; i <= numWorkers; i++) {
                            ACLMessage reply = new ACLMessage(ACLMessage.REQUEST);
                            reply.addReceiver(new AID("Worker" + i, AID.ISLOCALNAME));
                            reply.setContent(trainingConfig);
                            send(reply);
                        }
                    }
                } else {
                    block();
                }
            }
        });
    }

    private void requestDataSplit() {
        ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
        msg.addReceiver(new AID("DataDistributor", AID.ISLOCALNAME));
        msg.setContent("Split Data for " + numWorkers + " workers");
        send(msg);
        System.out.println(getLocalName() + " requested data splitting.");
    }

    private void notifyWorkersToPrepare() {
        for (int i = 1; i <= numWorkers; i++) {
            ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
            msg.addReceiver(new AID("Worker" + i, AID.ISLOCALNAME));
            msg.setContent("Prepare for Training");
            send(msg);
        }
        System.out.println(getLocalName() + " notified workers to prepare.");
    }

    private void requestDataDistribution() {
        ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
        msg.addReceiver(new AID("DataDistributor", AID.ISLOCALNAME));
        msg.setContent("Distribute The Data");
        send(msg);
        System.out.println(getLocalName() + " requested data distribution.");
    }

    private void notifyAggregatorToPrepare() {
        ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
        msg.addReceiver(new AID("ModelAggregator", AID.ISLOCALNAME));
        msg.setContent("Prepare for Aggregation");
        send(msg);
        System.out.println(getLocalName() + " notified ModelAggregatorAgent to prepare.");
    }

    private void notifyWorkersToSendModels() {
        for (int i = 1; i <= numWorkers; i++) {
            ACLMessage msg = new ACLMessage(ACLMessage.REQUEST);
            msg.addReceiver(new AID("Worker" + i, AID.ISLOCALNAME));
            msg.setContent("Send Model to Aggregator");
            send(msg);
        }
        System.out.println(getLocalName() + " notified workers to send models.");
    }
}