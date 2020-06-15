pipeline {
    agent { docker {image 'python:3.7-alpine'} }
    stages {
        stage('Init') {
            steps {
                echo 'Initialize requirements'
                sh 'pip intall -r requirements.txt'
            }
        }
        stage('Build') {
            steps {
                echo 'Building a docker image...'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Model Deployed clc, azure,...'
            }
        }
    }
}