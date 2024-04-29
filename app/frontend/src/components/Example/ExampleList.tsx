import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = ["Qual é o prazo de validade do contrato mais recente?", "Existe alguma cláusula de rescisão ou penalidade no contrato 106/2024 ?", "Quem são as partes envolvidas no contrato 106/2024 ?"];

const GPT4V_EXAMPLES: string[] = [
    "Existe alguma cláusula de arbitragem para resolver disputas?",
    "O que diz o parágrafo quinto do contrato 106/2024?",
    "Faça um resumo do contrato 106/2024"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
