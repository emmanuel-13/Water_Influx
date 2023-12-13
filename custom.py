custom_css = '''
           <style>
               .animate{
                   border-radius: 20px;
                   height: 300px;
                   width: 350px;
                   margin: 30px;
               }

               .text-content{
                   text-align: center;
               }

               .my_text {
                   color: red;
                   background-color: grey;
                   font-weight: bold;
                   text-align: center;
                   animation: my_change 10s infinite;
               }

               @keyframes my_change {
                   0% {color: red; background-color: green; }
                   50% {color: yellow; background-color: blue; }
                   100% {color: green; background-color: orange; }
               }

               .carousel-container {
                    display: flex;
                    overflow: hidden;
                }

                .carousel-item {
                    min-width: 100%;
                    transition: transform 0.3s ease;
                }

                .carousel-container {
                    display: flex;
                    overflow: hidden;
                    width: 400px; /* Set your preferred carousel width */
                }

                .carousel-item {
                    min-width: 400px; /* Set your preferred carousel width */
                    animation: slide 80s infinite; /* Adjust the slide duration as needed */
                }

                @keyframes slide {
                    0% { transform: translateX(0); }
                    25% { transform: translateX(-100%); }
                    50% { transform: translateX(-200%); }
                    75% { transform: translateX(-300%); }
                    100% { transform: translateX(0); }
                }

                #popin {
                    animation: popIn o.5 ease
                }

                @keyframes popIn {
                    0% {
                    opacity: 0;
                    transform: scale(0);
                    }

                    50% {
                        transform: scale(1.2);
                        opacity: 1;
                    }
                    %100 {
                        opacity: 1:
                        transform: scale(1);
                    }
                }
        </style>
'''